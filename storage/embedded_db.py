"""
嵌入式数据库与存储引擎 - 核心模块

功能：
- B树索引：节点分裂/合并、等值查找、磁盘持久化（checkpoint 时落盘，重启加载）
- 事务管理：原子性（回滚/崩溃恢复丢弃未提交操作）、隔离性（事务间互不可见
  未提交数据）、持久性（提交记录先 fsync 落盘）
- WAL 预写日志：提交前写日志并刷盘、检查点截断、崩溃后按 WAL 重做已提交事务
- 查询优化器：主键直查、依据索引与数据分布（基数/选择率）选择索引扫描或全表扫描
- 并发控制：记录级 S/X 锁、等待图（wait-for graph）死锁检测

仅使用 Python 标准库。
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import bisect
import json
import os
import threading
import time


class DeadlockError(Exception):
    """等待图检测到环，被选为牺牲者的事务被中止。"""


class LockTimeoutError(Exception):
    """等待锁超时。"""


@dataclass
class Config:
    data_dir: str = ".embedded_db_data"
    btree_degree: int = 32
    sync_commit: bool = True
    lock_timeout: float = 5.0
    index_selectivity_threshold: float = 0.3


# ---------------------------------------------------------------------------
# B 树索引（最小度数 t：节点键数范围 [t-1, 2t-1]）
# ---------------------------------------------------------------------------

class _BTreeNode:
    __slots__ = ("keys", "values", "children", "leaf")

    def __init__(self, leaf: bool = True):
        self.keys: List[Any] = []
        # 与 keys 平行：每个键对应一个主键列表（索引列允许重复值）
        self.values: List[List[Any]] = []
        self.children: List["_BTreeNode"] = []
        self.leaf = leaf

    def to_dict(self) -> dict:
        data = {"keys": list(self.keys),
                "values": [list(v) for v in self.values],
                "leaf": self.leaf}
        if not self.leaf:
            data["children"] = [c.to_dict() for c in self.children]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "_BTreeNode":
        node = cls(leaf=data["leaf"])
        node.keys = list(data["keys"])
        node.values = [list(v) for v in data["values"]]
        if not node.leaf:
            node.children = [cls.from_dict(c) for c in data["children"]]
        return node


def _iter_nodes(node: _BTreeNode):
    yield node
    for child in node.children:
        yield from _iter_nodes(child)


class BTree:
    """支持分裂（insert）与合并/借位（delete）、可序列化持久化的 B 树。"""

    def __init__(self, degree: int = 32):
        self.t = max(2, int(degree))
        self.root = _BTreeNode(leaf=True)
        self._key_count = 0

    @property
    def max_keys(self) -> int:
        return 2 * self.t - 1

    def key_count(self) -> int:
        """不同键的数量（用于基数/选择率统计）。"""
        return self._key_count

    # --- 查询 -------------------------------------------------------------

    def search(self, key: Any) -> Optional[List[Any]]:
        return self._search(self.root, key)

    def _search(self, node: _BTreeNode, key: Any):
        i = bisect.bisect_left(node.keys, key)
        if i < len(node.keys) and node.keys[i] == key:
            return node.values[i]
        if node.leaf:
            return None
        return self._search(node.children[i], key)

    def _find_node(self, node: _BTreeNode, key: Any):
        i = bisect.bisect_left(node.keys, key)
        if i < len(node.keys) and node.keys[i] == key:
            return node, i
        if node.leaf:
            return None
        return self._find_node(node.children[i], key)

    def items(self) -> List[Tuple[Any, List[Any]]]:
        out: List[Tuple[Any, List[Any]]] = []
        self._in_order(self.root, out)
        return out

    def _in_order(self, node: _BTreeNode, out):
        if node.leaf:
            for k, v in zip(node.keys, node.values):
                out.append((k, v))
            return
        for i, child in enumerate(node.children):
            self._in_order(child, out)
            if i < len(node.keys):
                out.append((node.keys[i], node.values[i]))

    def height(self) -> int:
        h = 1
        node = self.root
        while not node.leaf:
            node = node.children[0]
            h += 1
        return h

    def node_count(self) -> int:
        return sum(1 for _ in _iter_nodes(self.root))

    # --- 插入（分裂） -----------------------------------------------------

    def insert(self, key: Any, pk: Any) -> None:
        existing = self.search(key)
        if existing is not None:
            if pk not in existing:
                existing.append(pk)
            return
        root = self.root
        if len(root.keys) == self.max_keys:
            new_root = _BTreeNode(leaf=False)
            new_root.children = [root]
            self.root = new_root
            self._split_child(new_root, 0)
        self._insert_nonfull(self.root, key, pk)
        self._key_count += 1

    def _split_child(self, parent: _BTreeNode, i: int) -> None:
        t = self.t
        full = parent.children[i]
        new_node = _BTreeNode(leaf=full.leaf)
        mid_key = full.keys[t - 1]
        mid_val = full.values[t - 1]
        new_node.keys = full.keys[t:]
        new_node.values = full.values[t:]
        full.keys = full.keys[:t - 1]
        full.values = full.values[:t - 1]
        if not full.leaf:
            new_node.children = full.children[t:]
            full.children = full.children[:t]
        parent.keys.insert(i, mid_key)
        parent.values.insert(i, mid_val)
        parent.children.insert(i + 1, new_node)

    def _insert_nonfull(self, node: _BTreeNode, key: Any, pk: Any) -> None:
        i = bisect.bisect_left(node.keys, key)
        if node.leaf:
            node.keys.insert(i, key)
            node.values.insert(i, [pk])
            return
        child = node.children[i]
        if len(child.keys) == self.max_keys:
            self._split_child(node, i)
            if key > node.keys[i]:
                i += 1
            child = node.children[i]
        self._insert_nonfull(child, key, pk)

    # --- 删除（借位/合并） -------------------------------------------------

    def delete(self, key: Any, pk: Any = None) -> None:
        found = self._find_node(self.root, key)
        if found is None:
            return
        node, idx = found
        if pk is not None and len(node.values[idx]) > 1:
            if pk in node.values[idx]:
                node.values[idx].remove(pk)
            return
        self._delete_key(self.root, key)
        self._key_count -= 1
        if not self.root.leaf and len(self.root.keys) == 0:
            self.root = self.root.children[0]

    def _delete_key(self, node: _BTreeNode, key: Any) -> None:
        t = self.t
        i = bisect.bisect_left(node.keys, key)
        if i < len(node.keys) and node.keys[i] == key:
            if node.leaf:
                node.keys.pop(i)
                node.values.pop(i)
                return
            left, right = node.children[i], node.children[i + 1]
            if len(left.keys) >= t:
                pred_key, pred_val = self._max_pair(left)
                node.keys[i] = pred_key
                node.values[i] = pred_val
                self._delete_key(left, pred_key)
            elif len(right.keys) >= t:
                succ_key, succ_val = self._min_pair(right)
                node.keys[i] = succ_key
                node.values[i] = succ_val
                self._delete_key(right, succ_key)
            else:
                self._merge(node, i)
                self._delete_key(left, key)
            return
        if node.leaf:
            return
        at_end = i == len(node.keys)
        if len(node.children[i].keys) < t:
            self._fill(node, i)
        child = node.children[i - 1] if at_end and i > len(node.keys) else node.children[i]
        self._delete_key(child, key)

    def _fill(self, node: _BTreeNode, i: int) -> None:
        if i > 0 and len(node.children[i - 1].keys) >= self.t:
            self._borrow_from_prev(node, i)
        elif i < len(node.children) - 1 and len(node.children[i + 1].keys) >= self.t:
            self._borrow_from_next(node, i)
        elif i < len(node.children) - 1:
            self._merge(node, i)
        else:
            self._merge(node, i - 1)

    def _borrow_from_prev(self, node: _BTreeNode, i: int) -> None:
        child, sib = node.children[i], node.children[i - 1]
        child.keys.insert(0, node.keys[i - 1])
        child.values.insert(0, node.values[i - 1])
        node.keys[i - 1] = sib.keys.pop()
        node.values[i - 1] = sib.values.pop()
        if not child.leaf:
            child.children.insert(0, sib.children.pop())

    def _borrow_from_next(self, node: _BTreeNode, i: int) -> None:
        child, sib = node.children[i], node.children[i + 1]
        child.keys.append(node.keys[i])
        child.values.append(node.values[i])
        node.keys[i] = sib.keys.pop(0)
        node.values[i] = sib.values.pop(0)
        if not child.leaf:
            child.children.append(sib.children.pop(0))

    def _merge(self, node: _BTreeNode, i: int) -> None:
        left, right = node.children[i], node.children[i + 1]
        left.keys.append(node.keys.pop(i))
        left.values.append(node.values.pop(i))
        left.keys.extend(right.keys)
        left.values.extend(right.values)
        if not left.leaf:
            left.children.extend(right.children)
        node.children.pop(i + 1)

    def _max_pair(self, node: _BTreeNode):
        while not node.leaf:
            node = node.children[-1]
        return node.keys[-1], node.values[-1]

    def _min_pair(self, node: _BTreeNode):
        while not node.leaf:
            node = node.children[0]
        return node.keys[0], node.values[0]

    # --- 持久化 -----------------------------------------------------------

    def to_dict(self) -> dict:
        return {"degree": self.t, "root": self.root.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "BTree":
        tree = cls(degree=data.get("degree", 32))
        tree.root = _BTreeNode.from_dict(data["root"])
        tree._key_count = sum(len(n.keys) for n in _iter_nodes(tree.root))
        return tree


# ---------------------------------------------------------------------------
# 并发控制：记录级 S/X 锁 + 等待图死锁检测
# ---------------------------------------------------------------------------

class _LockManager:
    def __init__(self, timeout: float):
        self._timeout = timeout
        self._cond = threading.Condition()
        self._locks: Dict[Any, Dict[str, set]] = {}
        self._wait_for: Dict[int, set] = {}
        self._aborted: set = set()

    def _can_grant(self, txn: int, key: Any, mode: str) -> bool:
        entry = self._locks.get(key)
        if entry is None:
            return True
        if mode == "S":
            return not entry["X"] or entry["X"] == {txn}
        holders = entry["S"] | entry["X"]
        return not holders or holders <= {txn}

    def _deadlocked(self, start: int) -> bool:
        stack = [start]
        seen = {start}
        while stack:
            for blocker in self._wait_for.get(stack.pop(), ()):
                if blocker == start:
                    return True
                if blocker not in seen:
                    seen.add(blocker)
                    stack.append(blocker)
        return False

    def acquire(self, txn: int, key: Any, mode: str) -> None:
        deadline = time.monotonic() + self._timeout
        with self._cond:
            while not self._can_grant(txn, key, mode):
                entry = self._locks.get(key, {"S": set(), "X": set()})
                blockers = (entry["S"] | entry["X"]) - {txn}
                self._wait_for[txn] = set(blockers)
                if self._deadlocked(txn):
                    self._wait_for.pop(txn, None)
                    # 牺牲者被中止：立即释放它已经持有的锁并唤醒等待者，
                    # 否则另一方仍会被它阻塞，只能等到锁超时。
                    for held in list(self._locks.keys()):
                        ventry = self._locks[held]
                        ventry["S"].discard(txn)
                        ventry["X"].discard(txn)
                        if not ventry["S"] and not ventry["X"]:
                            del self._locks[held]
                    for blockers_set in self._wait_for.values():
                        blockers_set.discard(txn)
                    self._aborted.add(txn)
                    self._cond.notify_all()
                    raise DeadlockError(
                        f"txn {txn} 等待 {key!r} 时检测到死锁，事务被中止")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._wait_for.pop(txn, None)
                    raise LockTimeoutError(f"txn {txn} 等待锁 {key!r} 超时")
                self._cond.wait(min(remaining, 0.05))
            self._wait_for.pop(txn, None)
            entry = self._locks.setdefault(key, {"S": set(), "X": set()})
            if mode == "X":
                entry["S"].discard(txn)  # S -> X 升级
                entry["X"].add(txn)
            else:
                entry["S"].add(txn)

    def release_all(self, txn: int) -> None:
        with self._cond:
            self._aborted.discard(txn)
            for key in list(self._locks.keys()):
                entry = self._locks[key]
                entry["S"].discard(txn)
                entry["X"].discard(txn)
                if not entry["S"] and not entry["X"]:
                    del self._locks[key]
            self._wait_for.pop(txn, None)
            for blockers in self._wait_for.values():
                blockers.discard(txn)
            self._cond.notify_all()

    def is_aborted(self, txn: int) -> bool:
        with self._cond:
            return txn in self._aborted


# ---------------------------------------------------------------------------
# 事务
# ---------------------------------------------------------------------------

class _Transaction:
    __slots__ = ("tid", "state", "ops")

    def __init__(self, tid: int):
        self.tid = tid
        self.state = "active"          # active / committed / rolled_back
        self.ops: List[tuple] = []     # ("insert"/"update"/"delete", table, ...)


# ---------------------------------------------------------------------------
# 嵌入式数据库
# ---------------------------------------------------------------------------

class EmbeddedDB:
    def __init__(self, config: Optional[Dict] = None):
        defaults = Config()
        self.config = dict(config or {})
        self.data_dir = self.config.get("data_dir", defaults.data_dir)
        self._degree = int(self.config.get("btree_degree", defaults.btree_degree))
        self._sync_commit = bool(
            self.config.get("sync_commit", defaults.sync_commit))
        self._lock_timeout = float(
            self.config.get("lock_timeout", defaults.lock_timeout))
        self._selectivity_threshold = float(
            self.config.get("index_selectivity_threshold",
                           defaults.index_selectivity_threshold))

        os.makedirs(self.data_dir, exist_ok=True)
        self._index_dir = os.path.join(self.data_dir, "indexes")
        os.makedirs(self._index_dir, exist_ok=True)
        self._wal_path = os.path.join(self.data_dir, "wal.log")
        self._checkpoint_path = os.path.join(self.data_dir, "checkpoint.json")

        self._meta_lock = threading.RLock()
        self._wal_lock = threading.Lock()
        self._lock_manager = _LockManager(self._lock_timeout)

        self._tables: Dict[str, dict] = {}
        self._txns: Dict[int, _Transaction] = {}
        self._txn_seq = 0
        self._closed = False

        self.stats = {"full_scans": 0, "index_scans": 0, "pk_lookups": 0}
        # 保持既有 API：_state / _history
        self._state: Dict[str, Any] = {}
        self._history: List[dict] = []

        self._load_checkpoint()
        self._wal_file = open(self._wal_path, "a", encoding="utf-8")
        self._recover()

    # 既有游戏循环 API -------------------------------------------------------

    def update(self, dt: float) -> None:
        pass

    def reset(self) -> None:
        self._state = {}
        self._history = []

    def close(self) -> None:
        if self._closed:
            return
        self.checkpoint()
        with self._wal_lock:
            self._wal_file.close()
        self._closed = True

    def crash(self) -> None:
        """模拟进程崩溃：不做检查点，直接丢弃全部内存状态（已提交数据靠 WAL 存活）。"""
        if self._closed:
            return
        with self._wal_lock:
            try:
                self._wal_file.close()
            except Exception:
                pass
        self._closed = True
        self._tables = {}
        self._txns = {}

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("数据库已关闭")

    # --- DDL ---------------------------------------------------------------

    def create_table(self, table: str, primary_key: str = "id") -> None:
        with self._meta_lock:
            self._ensure_open()
            if table in self._tables:
                raise ValueError(f"表 {table!r} 已存在")
            self._tables[table] = {"pk": primary_key, "rows": {}, "indexes": {}}
            self._wal_append(
                {"type": "create_table", "table": table, "pk": primary_key},
                sync=True)

    def create_index(self, table: str, column: str) -> BTree:
        with self._meta_lock:
            self._ensure_open()
            tab = self._tables[table]
            if column in tab["indexes"]:
                return tab["indexes"][column]
            tree = BTree(degree=self._degree)
            pk_col = tab["pk"]
            for row in tab["rows"].values():
                tree.insert(row[column], row[pk_col])
            tab["indexes"][column] = tree
            self._wal_append(
                {"type": "create_index", "table": table, "column": column},
                sync=True)
            return tree

    def index_info(self, table: str, column: str) -> dict:
        tab = self._tables[table]
        tree = tab["indexes"][column]
        return {"height": tree.height(),
                "nodes": tree.node_count(),
                "distinct_keys": tree.key_count(),
                "entries": sum(len(v) for _, v in tree.items())}

    def row_count(self, table: str) -> int:
        return len(self._tables[table]["rows"])

    # --- WAL ---------------------------------------------------------------

    def _wal_append(self, record: dict, sync: bool = False) -> None:
        with self._wal_lock:
            self._wal_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._wal_file.flush()
            if sync or self._sync_commit and record.get("type") == "commit":
                os.fsync(self._wal_file.fileno())

    def _recover(self) -> None:
        """重放 WAL：只重做带 COMMIT 记录的事务，未提交操作全部丢弃（原子性）。

        崩溃可能在日志末尾留下半行（部分写入），重放到该处停止，并把文件
        截断到最后一条完整记录的边界，避免之后追加的记录与残行粘在一行。
        """
        if not os.path.exists(self._wal_path):
            return
        pending: Dict[int, List[dict]] = {}
        valid_len = 0
        max_txn = 0
        with open(self._wal_path, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    valid_len = f.tell()
                    continue
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError:
                    break  # 崩溃时可能留下半行，停止重放
                valid_len = f.tell()
                rtype = rec.get("type")
                if "txn" in rec:
                    max_txn = max(max_txn, int(rec["txn"]))
                if rtype == "create_table":
                    if rec["table"] not in self._tables:
                        self._tables[rec["table"]] = {
                            "pk": rec["pk"], "rows": {}, "indexes": {}}
                elif rtype == "create_index":
                    tab = self._tables.get(rec["table"])
                    if tab is not None and rec["column"] not in tab["indexes"]:
                        tree = BTree(degree=self._degree)
                        for row in tab["rows"].values():
                            tree.insert(row[rec["column"]], row[tab["pk"]])
                        tab["indexes"][rec["column"]] = tree
                elif rtype == "op":
                    pending.setdefault(rec["txn"], []).append(rec)
                elif rtype == "commit":
                    for op in pending.pop(rec["txn"], []):
                        self._apply_wal_op(op)
                elif rtype == "rollback":
                    pending.pop(rec["txn"], None)
                elif rtype == "checkpoint":
                    pending.clear()
        file_size = os.path.getsize(self._wal_path)
        if valid_len < file_size:
            with open(self._wal_path, "r+b") as f:
                f.truncate(valid_len)
        # 重启后事务号必须严格递增，否则新事务可能复用 WAL 中残留的
        # 未提交事务号，导致恢复时把未提交操作一并重放。
        self._txn_seq = max(self._txn_seq, max_txn)

    def _apply_wal_op(self, rec: dict) -> None:
        kind = rec["op"]
        if kind == "insert":
            self._apply_op(("insert", rec["table"], rec["row"]))
        elif kind == "update":
            self._apply_op(("update", rec["table"], rec["pk"], rec["changes"]))
        elif kind == "delete":
            self._apply_op(("delete", rec["table"], rec["pk"]))

    def checkpoint(self) -> None:
        """把表数据与 B 树索引落盘，然后截断 WAL（日志不再无限增长）。"""
        with self._meta_lock:
            self._ensure_open()
            data = {"tables": {}}
            for name, tab in self._tables.items():
                data["tables"][name] = {
                    "pk": tab["pk"],
                    "rows": list(tab["rows"].values()),
                    "indexes": sorted(tab["indexes"].keys()),
                }
            tmp = self._checkpoint_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._checkpoint_path)
            # 索引单独持久化
            for name, tab in self._tables.items():
                for column, tree in tab["indexes"].items():
                    self._save_index(name, column, tree)
            # 截断 WAL
            with self._wal_lock:
                self._wal_file.close()
                self._wal_file = open(self._wal_path, "w", encoding="utf-8")
            self._wal_append({"type": "checkpoint"}, sync=True)

    def _index_path(self, table: str, column: str) -> str:
        return os.path.join(self._index_dir, f"{table}__{column}.json")

    def _save_index(self, table: str, column: str, tree: BTree) -> None:
        path = self._index_path(table, column)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(tree.to_dict(), f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _load_checkpoint(self) -> None:
        if not os.path.exists(self._checkpoint_path):
            return
        with open(self._checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name, tdata in data.get("tables", {}).items():
            pk_col = tdata["pk"]
            rows = {row[pk_col]: row for row in tdata["rows"]}
            tab = {"pk": pk_col, "rows": rows, "indexes": {}}
            for column in tdata.get("indexes", []):
                path = self._index_path(name, column)
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        tab["indexes"][column] = BTree.from_dict(json.load(f))
                else:
                    # 索引文件缺失时从数据重建（兜底）
                    tree = BTree(degree=self._degree)
                    for row in rows.values():
                        tree.insert(row[column], row[pk_col])
                    tab["indexes"][column] = tree
            self._tables[name] = tab

    # --- 事务管理 -----------------------------------------------------------

    def begin(self) -> int:
        self._ensure_open()
        with self._meta_lock:
            self._txn_seq += 1
            tid = self._txn_seq
            self._txns[tid] = _Transaction(tid)
        self._wal_append({"type": "begin", "txn": tid})
        return tid

    def _get_txn(self, tid: int) -> _Transaction:
        txn = self._txns.get(tid)
        if txn is None or txn.state != "active":
            raise ValueError(f"事务 {tid} 不存在或已结束")
        if self._lock_manager.is_aborted(tid):
            raise DeadlockError(f"事务 {tid} 已因死锁被中止，请回滚后重试")
        return txn

    def commit(self, tid: int) -> None:
        txn = self._get_txn(tid)
        # 持久性：COMMIT 记录先写 WAL 并 fsync，再应用到内存表
        self._wal_append({"type": "commit", "txn": tid}, sync=True)
        with self._meta_lock:
            for op in txn.ops:
                self._apply_op(op)
            txn.state = "committed"
            self._txns.pop(tid, None)
        self._lock_manager.release_all(tid)

    def rollback(self, tid: int) -> None:
        txn = self._txns.get(tid)
        if txn is None or txn.state != "active":
            raise ValueError(f"事务 {tid} 不存在或已结束")
        # 死锁牺牲者的锁在中止时已释放，这里允许正常回滚清理事务状态
        self._wal_append({"type": "rollback", "txn": tid})
        txn.ops.clear()
        txn.state = "rolled_back"
        self._txns.pop(tid, None)
        self._lock_manager.release_all(tid)

    def _apply_op(self, op: tuple) -> None:
        kind, table = op[0], op[1]
        tab = self._tables[table]
        pk_col = tab["pk"]
        if kind == "insert":
            row = dict(op[2])
            tab["rows"][row[pk_col]] = row
            for column, tree in tab["indexes"].items():
                tree.insert(row[column], row[pk_col])
        elif kind == "update":
            pk, changes = op[2], op[3]
            old = tab["rows"].get(pk)
            if old is None:
                return
            new_row = {**old, **changes}
            tab["rows"][pk] = new_row
            for column, tree in tab["indexes"].items():
                if column in changes and changes[column] != old[column]:
                    tree.delete(old[column], pk)
                    tree.insert(new_row[column], pk)
        elif kind == "delete":
            pk = op[2]
            old = tab["rows"].pop(pk, None)
            if old is None:
                return
            for column, tree in tab["indexes"].items():
                tree.delete(old[column], pk)

    # --- DML ---------------------------------------------------------------

    def insert(self, table: str, row: Dict[str, Any], txn: Optional[int] = None) -> None:
        if txn is None:
            tid = self.begin()
            try:
                self._insert_op(table, row, tid)
                self.commit(tid)
            except Exception:
                self._quiet_rollback(tid)
                raise
        else:
            self._insert_op(table, row, txn)

    def update_row(self, table: str, pk: Any, changes: Dict[str, Any],
                   txn: Optional[int] = None) -> None:
        if txn is None:
            tid = self.begin()
            try:
                self._update_op(table, pk, changes, tid)
                self.commit(tid)
            except Exception:
                self._quiet_rollback(tid)
                raise
        else:
            self._update_op(table, pk, changes, txn)

    def delete_row(self, table: str, pk: Any, txn: Optional[int] = None) -> None:
        if txn is None:
            tid = self.begin()
            try:
                self._delete_op(table, pk, tid)
                self.commit(tid)
            except Exception:
                self._quiet_rollback(tid)
                raise
        else:
            self._delete_op(table, pk, txn)

    def _quiet_rollback(self, tid: int) -> None:
        try:
            if tid in self._txns:
                self.rollback(tid)
        except Exception:
            pass

    def _insert_op(self, table: str, row: Dict[str, Any], tid: int) -> None:
        self._ensure_open()
        txn = self._get_txn(tid)
        tab = self._tables[table]
        pk_col = tab["pk"]
        pk = row[pk_col]
        self._lock_manager.acquire(tid, (table, pk), "X")
        if self._row_visible(table, pk, txn) is not None:
            raise ValueError(f"主键冲突：{table}.{pk_col}={pk!r} 已存在")
        txn.ops.append(("insert", table, dict(row)))
        self._wal_append({"type": "op", "txn": tid, "op": "insert",
                          "table": table, "row": dict(row)})

    def _update_op(self, table: str, pk: Any, changes: Dict[str, Any],
                   tid: int) -> None:
        self._ensure_open()
        txn = self._get_txn(tid)
        self._lock_manager.acquire(tid, (table, pk), "X")
        if self._row_visible(table, pk, txn) is None:
            raise KeyError(f"记录不存在：{table} pk={pk!r}")
        txn.ops.append(("update", table, pk, dict(changes)))
        self._wal_append({"type": "op", "txn": tid, "op": "update",
                          "table": table, "pk": pk, "changes": dict(changes)})

    def _delete_op(self, table: str, pk: Any, tid: int) -> None:
        self._ensure_open()
        txn = self._get_txn(tid)
        self._lock_manager.acquire(tid, (table, pk), "X")
        if self._row_visible(table, pk, txn) is None:
            raise KeyError(f"记录不存在：{table} pk={pk!r}")
        txn.ops.append(("delete", table, pk))
        self._wal_append({"type": "op", "txn": tid, "op": "delete",
                          "table": table, "pk": pk})

    # --- 查询 ---------------------------------------------------------------

    def get(self, table: str, pk: Any, txn: Optional[int] = None,
            for_update: bool = False) -> Optional[Dict[str, Any]]:
        self._ensure_open()
        tab = self._tables[table]
        if txn is not None:
            txn_obj = self._get_txn(txn)
            self._lock_manager.acquire(txn, (table, pk),
                                       "X" if for_update else "S")
            row = self._row_visible(table, pk, txn_obj)
        else:
            # 自动提交读：只读已提交数据（读已提交隔离级别）
            row = tab["rows"].get(pk)
        return dict(row) if row is not None else None

    def _row_visible(self, table: str, pk: Any,
                     txn: Optional[_Transaction]) -> Optional[Dict[str, Any]]:
        tab = self._tables[table]
        row = tab["rows"].get(pk)
        row = dict(row) if row is not None else None
        if txn is None:
            return row
        pk_col = tab["pk"]
        for op in txn.ops:
            if op[1] != table:
                continue
            if op[0] == "insert" and op[2][pk_col] == pk:
                row = dict(op[2])
            elif op[0] == "update" and op[2] == pk and row is not None:
                row.update(op[3])
            elif op[0] == "delete" and op[2] == pk:
                row = None
        return row

    def query(self, table: str, where: Optional[Dict[str, Any]] = None,
              txn: Optional[int] = None) -> List[Dict[str, Any]]:
        self._ensure_open()
        tab = self._tables[table]
        where = dict(where or {})
        plan = self.explain(table, where)
        self._history.append(plan)
        self.stats[{"full_scan": "full_scans",
                    "index_scan": "index_scans",
                    "pk_lookup": "pk_lookups"}[plan["type"]]] += 1

        if plan["type"] == "pk_lookup":
            row = tab["rows"].get(where[tab["pk"]])
            candidates = [row] if row is not None else []
        elif plan["type"] == "index_scan":
            tree = tab["indexes"][plan["index_column"]]
            pks = tree.search(plan["value"]) or []
            candidates = [tab["rows"][pk] for pk in pks if pk in tab["rows"]]
        else:
            candidates = list(tab["rows"].values())

        rows = {r[tab["pk"]]: dict(r) for r in candidates
                if self._matches(r, where)}

        txn_obj = self._txns.get(txn) if txn is not None else None
        if txn_obj is not None and txn_obj.state == "active":
            rows = self._overlay_query(tab, table, rows, where, txn_obj)
        return list(rows.values())

    def _overlay_query(self, tab: dict, table: str,
                       rows: Dict[Any, dict], where: dict,
                       txn: _Transaction) -> Dict[Any, dict]:
        pk_col = tab["pk"]
        for op in txn.ops:
            if op[1] != table:
                continue
            if op[0] == "insert":
                row = dict(op[2])
                if self._matches(row, where):
                    rows[row[pk_col]] = row
            elif op[0] == "update":
                pk = op[2]
                if pk in rows:
                    rows[pk].update(op[3])
                    if not self._matches(rows[pk], where):
                        rows.pop(pk, None)
            elif op[0] == "delete":
                rows.pop(op[2], None)
        return rows

    @staticmethod
    def _matches(row: Dict[str, Any], where: Dict[str, Any]) -> bool:
        return all(row.get(col) == val for col, val in where.items())

    # --- 查询优化器 ----------------------------------------------------------

    def explain(self, table: str,
                where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """依据索引与数据分布（列基数/选择率）选择执行计划。"""
        tab = self._tables[table]
        where = dict(where or {})
        total = len(tab["rows"])
        plan: Dict[str, Any] = {"table": table, "where": where,
                                "total_rows": total}

        if not where:
            plan.update(type="full_scan", reason="无过滤条件")
            return plan

        if tab["pk"] in where:
            plan.update(type="pk_lookup", estimated_rows=1,
                        reason="主键等值条件，直接定位")
            return plan

        best = None
        for column, value in where.items():
            tree = tab["indexes"].get(column)
            if tree is None:
                continue
            distinct = max(tree.key_count(), 1)
            estimated = max(total / distinct, 1.0)
            selectivity = estimated / total if total else 1.0
            if best is None or estimated < best[2]:
                best = (column, value, estimated, selectivity)

        if best is not None and best[3] <= self._selectivity_threshold:
            plan.update(type="index_scan", index_column=best[0], value=best[1],
                        estimated_rows=best[2], selectivity=best[3],
                        reason=f"索引 {best[0]} 选择率 {best[3]:.4f} 足够低")
            return plan

        if best is not None:
            plan.update(type="full_scan", estimated_rows=total,
                        selectivity=best[3],
                        reason=f"最佳索引选择率 {best[3]:.4f} 过高，"
                               "索引扫描不划算，退化为全表扫描")
        else:
            plan.update(type="full_scan", estimated_rows=total,
                        reason="条件列无可用索引")
        return plan
