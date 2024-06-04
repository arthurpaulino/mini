use indexmap::IndexSet;
use std::hash::{BuildHasher, Hasher};

/// A hasher that only knows how to hash data of type `usize`, simply returning
/// it as `u64` for speed.
struct MemHasher {
    hash: usize,
}

impl Hasher for MemHasher {
    fn write(&mut self, _bytes: &[u8]) {
        panic!("Bytes aren't expected to be hashed.");
    }

    fn write_usize(&mut self, i: usize) {
        self.hash = i;
    }

    fn finish(&self) -> u64 {
        self.hash as u64
    }
}

///A builder for `MemHasher`.
#[derive(Clone, Default)]
struct BuildMemHasher;

impl BuildHasher for BuildMemHasher {
    type Hasher = MemHasher;

    fn build_hasher(&self) -> Self::Hasher {
        MemHasher { hash: 0 }
    }
}

///A builder for `MemHasher`.
#[derive(Clone, Default)]
struct Mem<T> {
    data: Vec<T>,
    available: IndexSet<usize, BuildMemHasher>,
}

impl<T> Mem<T> {
    /// Creates a `Mem` from a given data vector.
    fn raw(data: Vec<T>) -> Self {
        Self {
            data,
            available: IndexSet::with_hasher(BuildMemHasher),
        }
    }
    
    /// Creates a `Mem` with specified capacity.
    fn with_capacity(n: usize) -> Self {
        Mem::raw(Vec::with_capacity(n))
    }

    /// Allocates an item and returns it's address;
    fn intern(&mut self, t: T) -> usize {
        if let Some(addr) = self.available.pop() {
            self.data[addr] = t;
            addr
        } else {
            let addr = self.data.len();
            self.data.push(t);
            addr
        }
    }

    /// Gets a reference to an item by it's address.
    fn get(&self, addr: usize) -> &T {
        &self.data[addr]
    }

    /// Gets a mutable reference to an item by it's address.
    fn get_mut(&mut self, addr: usize) -> &mut T {
        &mut self.data[addr]
    }
    
    /// Frees and alocated slot.
    fn free(&mut self, addr: usize) {
        self.available.insert(addr);
    }

    /// Checks if a slot if not free.
    fn not_free(&self, addr: &usize) -> bool {
        !self.available.contains(addr)
    }
}

/// Represents a node in a circular doubly linked list.
#[derive(Clone, Default)]
struct Node<T> {
    data: Option<T>,
    prev: usize,
    next: usize,
}

impl<T> Node<T> {
    
    /// Creates a root node with no data.
    fn root() -> Self {
        Self {
            data: None,
            prev: 0,
            next: 0,
        }
    }

    /// Initializes a node with data.
    fn init(t: T) -> Self {
        Self {
            data: Some(t),
            prev: 0,
            next: 0,
        }
    }
}

/// A circular doubly linked list with custom memory management.
#[derive(Clone, Default)]
pub struct CList<T> {
    mem: Mem<Node<T>>,
    len: usize,
}

pub const ROOT_ADDR: usize = 0;

impl<T> CList<T> {
    
    /// Creates a list with specified capacity.
    ///
    /// # Arguments
    ///
    /// * `n` - The capacity of the list.
    ///
    /// # Returns 
    ///
    /// A  `CList` with the specified capacity.
    #[inline]
    pub fn with_capacity(n: usize) -> Self {
        let mut mem = Mem::with_capacity(n + 1);
        assert_eq!(mem.intern(Node::root()), ROOT_ADDR);
        Self { mem, len: 0 }
    }

    /// Creates a list from a slice. 
    ///
    /// # Arguments
    ///
    /// * `slice` - A slice of elements. 
    ///
    /// # Returns 
    ///
    /// A  `CList` containing the elements of the slice. 
    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Clone,
    {
        let len = slice.len();
        let mut nodes = vec![Node::root(); len + 1];
        if len > 0 {
            nodes[0].next = ROOT_ADDR + 1;
            nodes[0].prev = len;
            nodes[1..].iter_mut().enumerate().for_each(|(i, node)| {
                node.data = Some(slice[i].clone());
                node.prev = i;
                node.next = if i == len - 1 { ROOT_ADDR } else { i + 2 };
            });
        }
        Self {
            mem: Mem::raw(nodes),
            len,
        }
    }

    /// Creates a list from a vector.
    ///
    /// # Arguments
    ///
    /// * `vec` - A vector of elements. 
    ///
    /// # Returns 
    ///
    /// A  `CList` containing the elements of the vector.
   pub fn from_vec(vec: Vec<T>) -> Self
    where
        T: Clone,
    {
        let len = vec.len();
        let mut nodes = vec![Node::root(); len + 1];
        if len > 0 {
            nodes[0].next = ROOT_ADDR + 1;
            nodes[0].prev = len;
            nodes[1..]
                .iter_mut()
                .enumerate()
                .zip(vec)
                .for_each(|((i, node), t)| {
                    node.data = Some(t);
                    node.prev = i;
                    node.next = if i == len - 1 { ROOT_ADDR } else { i + 2 };
                });
        }
        Self {
            mem: Mem::raw(nodes),
            len,
        }
    }

    /// Returns the number of elements in the list.
    ///
    /// # Returns 
    ///
    /// The number of elements in the list.
   #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Checks if the list is empty
    ///
    /// # Returns 
    ///
    /// `true` if the list is empty, `false` otherwise.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Asserts that the node at the given address is not free.
    ///
    /// # Arguments
    ///
    /// * `node_addr` - The address of the node.
    fn assert_not_free(&self, node_addr: &usize) {
        assert!(self.mem.not_free(node_addr), "Node address is free.");
    }

    /// Gets a reference to the data at the given node address.
    ///
    /// # Arguments
    ///
    /// * `node_addr` - the address of the node.
    ///
    /// # Returns
    ///
    /// A reference to the data at the node address.
   #[inline]
    pub fn get(&self, node_addr: usize) -> &T {
        self.assert_not_free(&node_addr);
        self.mem
            .get(node_addr)
            .data
            .as_ref()
            .expect("Address doesn't contain data.")
    }

    /// Gets a mutable reference to the data at the given node address.
    /// 
    /// # Arguments
    ///
    /// * `node_addr` - the address of the node.
    ///
    /// # Returns
    ///
    /// A mutable reference to the data at the node address.
    #[inline]
    pub fn get_mut(&mut self, node_addr: usize) -> &mut T {
        self.assert_not_free(&node_addr);
        self.mem
            .get_mut(node_addr)
            .data
            .as_mut()
            .expect("Address doesn't contain data.")
    }

    /// Gets the previous node address of the given node address.
    /// 
    /// # Arguments
    ///
    /// * `node_addr` - the address of the node.
    ///
    /// # Returns
    ///
    /// The previous node address.
   #[inline]
    pub fn prev(&self, node_addr: usize) -> usize {
        self.assert_not_free(&node_addr);
        self.mem.get(node_addr).prev
    }

    /// Gets the next node address of the given node address.
    /// 
    /// # Arguments
    ///
    /// * `node_addr` - the address of the node.
    ///
    /// # Returns
    ///
    /// The next node address.
   #[inline]
    pub fn next(&self, node_addr: usize) -> usize {
        self.assert_not_free(&node_addr);
        self.mem.get(node_addr).next
    }

    /// Gets the address of the first node.
    /// 
    /// # Arguments
    ///
    /// * `node_addr` - the address of the node.
    ///
    /// # Returns
    ///
    /// The address of the first node.
   #[inline]
    pub fn first_addr(&self) -> usize {
        self.next(ROOT_ADDR)
    }

    /// Gets the address of the last node.
    /// 
    /// # Arguments
    ///
    /// * `node_addr` - the address of the node.
    ///
    /// # Returns
    ///
    /// The address of the last node.
   #[inline]
    pub fn last_addr(&self) -> usize {
        self.prev(ROOT_ADDR)
    }

    /// Inserts a new node with the given data before or after a reference node.
    /// 
    /// # Arguments
    ///
    /// * `t` - the data to insert.
    /// * `ref_node_addr` - the address to the reference node.
    /// * `after` - Whether to insert after (`true`) or before (`false`) the reference node.
    ///
    /// # Returns
    ///
    /// The address of the new node.
   pub fn insert_ref(&mut self, t: T, ref_node_addr: usize, after: bool) -> usize {
        self.assert_not_free(&ref_node_addr);
        let new_node_addr = self.mem.intern(Node::init(t));
        let ref_node = self.mem.get_mut(ref_node_addr);
        let (prev, next) = if after {
            let ref_node_next = ref_node.next;
            ref_node.next = new_node_addr;
            let next_node = self.mem.get_mut(ref_node_next);
            let next_node_prev = next_node.prev;
            next_node.prev = new_node_addr;
            (next_node_prev, ref_node_next)
        } else {
            let ref_node_prev = ref_node.prev;
            ref_node.prev = new_node_addr;
            let prev_node = self.mem.get_mut(ref_node_prev);
            let prev_node_next = prev_node.next;
            prev_node.next = new_node_addr;
            (ref_node_prev, prev_node_next)
        };
        let node = self.mem.get_mut(new_node_addr);
        node.prev = prev;
        node.next = next;
        self.len += 1;
        new_node_addr
    }

    /// Removes a node at the given address.
    /// 
    /// # Arguments
    ///
    /// * `node_addr` - the address of the node to be removed.
    ///
    /// # Returns
    ///
    /// The address of the next node.
  pub fn remove(&mut self, node_addr: usize) -> usize {
        if node_addr == ROOT_ADDR {
            return node_addr;
        }
        self.assert_not_free(&node_addr);
        let ref_node = self.mem.get(node_addr);
        let (left, right) = (ref_node.prev, ref_node.next);
        self.mem.get_mut(left).next = right;
        self.mem.get_mut(right).prev = left;
        self.mem.free(node_addr);
        self.len -= 1;
        right
    }

    /// Removes at max `n` nodes starting from the given address.
    /// 
    /// # Arguments
    ///
    /// * `node_addr` - the starting address to the node to remove.
    /// * `n` - the number of nodes to remove.
    ///
    /// # Returns
    ///
    /// The address of the node after the last removed node.
   pub fn remove_n(&mut self, mut node_addr: usize, n: usize) -> usize {
        self.assert_not_free(&node_addr);
        let left = self.mem.get(node_addr).prev;
        let mut removed = 0;
        for _ in 0..n {
            if node_addr == ROOT_ADDR {
                break;
            }
            self.mem.free(node_addr);
            node_addr = self.mem.get(node_addr).next;
            removed += 1;
        }
        self.mem.get_mut(left).next = node_addr;
        self.mem.get_mut(node_addr).prev = left;
        self.len -= removed;
        node_addr
    }

    /// Inserts multiple nodes to the right of the reference node.
    /// 
    /// # Arguments
    ///
    /// * `ref_node_addr` - The address of the reference node.
    /// * `data` - An iterator of elements to insert.
    ///
    /// # Returns
    ///
    /// The address of the last inserted node.
   #[inline]
    pub fn insert_iter_right<I: IntoIterator<Item = T>>(
        &mut self,
        ref_node_addr: usize,
        data: I,
    ) -> usize {
        data.into_iter().fold(ref_node_addr, |node_addr, t| {
            self.insert_ref(t, node_addr, true)
        })
    }

    /// Inserts multiple nodes to the left of the reference node.
    /// 
    /// # Arguments
    ///
    /// * `ref_node_addr` - The address of the reference node.
    /// * `data` - An iterator of elements to insert.
    ///
    /// # Returns
    ///
    /// The address of the last inserted node.
   #[inline]
    pub fn insert_iter_left<I: IntoIterator<Item = T>>(
        &mut self,
        ref_node_addr: usize,
        data: I,
    ) -> usize
    where
        <I as IntoIterator>::IntoIter: DoubleEndedIterator,
    {
        data.into_iter().rev().fold(ref_node_addr, |node_addr, t| {
            self.insert_ref(t, node_addr, false)
        })
    }

    /// Fills a buffer with the data from the list starting from a given node address.
    /// 
    /// # Arguments
    ///
    /// * `buf` - The buffer to fill.
    /// * `node_addr` - The starting address of the node.
    /// * `data` - An iterator of elements to insert.
    ///
    /// # Returns
    ///
    /// The filled buffer.
   fn populate_buf(&self, mut buf: Vec<T>, mut node_addr: usize) -> Vec<T>
    where
        T: Clone,
    {
        loop {
            let node = self.mem.get(node_addr);
            if let Some(t) = &node.data {
                buf.push(t.clone());
            }
            if node.next == ROOT_ADDR {
                break;
            }
            node_addr = node.next;
        }
        buf
    }

    /// Collects all elements from the list into a vector.
    ///
    /// # Returns
    ///
    /// A vector containing all the elements in the list.
   #[inline]
    pub fn collect(&self) -> Vec<T>
    where
        T: Clone,
    {
        if self.is_empty() {
            return vec![];
        }
        self.populate_buf(Vec::with_capacity(self.len), ROOT_ADDR)
    }

    /// Collects all elements from the list, starting from a given address into a vector
    ///
    /// # Arguments
    ///
    /// * `node_addr` - The address of the starting node.
    /// 
    /// # Returns
    ///
    /// A vector containing all the elements from the starting address.
   #[inline]
    pub fn collect_from_addr(&self, node_addr: usize, capacity_hint: Option<usize>) -> Vec<T>
    where
        T: Clone,
    {
        if self.is_empty() {
            return vec![];
        }
        // If there's no capacity hint, we guess len/2.
        let capacity = capacity_hint.unwrap_or(self.len >> 1);
        self.populate_buf(Vec::with_capacity(capacity), node_addr)
    }

    /// Fills a buffer with the data from the list starting from a given node address.
    /// 
    /// # Arguments
    ///
    /// * `buf` - The buffer to fill.
    /// * `node_addr` - The starting address of the node.
    /// * `data` - An iterator of elements to insert.
    /// * `n` - Max number of elements to insert. 
    ///
    /// # Returns
    ///
    /// The filled buffer.
   fn populate_buf_n(&self, mut buf: Vec<T>, mut node_addr: usize, mut n: usize) -> Vec<T>
    where
        T: Clone,
    {
        loop {
            let node = self.mem.get(node_addr);
            if let Some(t) = &node.data {
                buf.push(t.clone());
                n -= 1;
            }
            if node.next == ROOT_ADDR || n == 0 {
                break;
            }
            node_addr = node.next;
        }
        buf
    }

    /// Collects n elements from the list into a vector.
    /// 
    /// # Arguments
    ///
    /// * `n` - Max number of elements to collect.
    ///
    /// # Returns
    ///
    /// A vector containing all the elements in the list.
   #[inline]
    pub fn collect_n(&self, n: usize) -> Vec<T>
    where
        T: Clone,
    {
        if self.is_empty() || n == 0 {
            return vec![];
        }
        self.populate_buf_n(Vec::with_capacity(n.min(self.len)), ROOT_ADDR, n)
    }

    /// Collects up to `n` elements, starting from a given position in the list
    ///
    /// If the list is empty, if `n` is zero, or if the position is out of bounds,
    /// it returns an empty vector.
    ///
    /// # Arguments
    ///
    /// * `n` - The the number of elements to collect.
    /// * `pos` - The starting position in the list.
    /// 
    /// # Returns
    ///
    /// A vector containing the collected elements.
    ///
    /// # Panics
    ///
    /// This function will panic if the list is modified during iteration.
   pub fn collect_n_from_pos(&self, n: usize, pos: usize) -> Vec<T>
    where
        T: Clone,
    {
        if self.is_empty() || n == 0 || pos >= self.len {
            return vec![];
        }
        let mut node_addr = ROOT_ADDR;
        let mut idx = 0;
        loop {
            node_addr = self.next(node_addr);
            if idx == pos {
                break;
            }
            idx += 1;
        }
        self.populate_buf_n(Vec::with_capacity(n.min(self.len - pos)), node_addr, n)
    }
}

#[cfg(test)]
mod tests {
    use super::{CList, ROOT_ADDR};

    impl<T> CList<T> {
        /// Slower version of `collect` meant for testing to assert that the `prev`
        /// values are consistent.
        fn collect_rev(&self) -> Vec<T>
        where
            T: Clone,
        {
            if self.is_empty() {
                return vec![];
            }
            let mut buf = Vec::with_capacity(self.len);
            let mut node_addr = ROOT_ADDR;
            loop {
                let node = self.mem.get(node_addr);
                if let Some(t) = &node.data {
                    buf.push(t.clone());
                }
                if node.prev == ROOT_ADDR {
                    break;
                }
                node_addr = node.prev;
            }
            buf.reverse();
            buf
        }
    }

    #[test]
    fn test_from_slice() {
        let assert_from_slice_roundtrip = |slice| {
            let clist = CList::from_slice(slice);
            assert_eq!(&clist.collect(), slice);
            assert_eq!(&clist.collect_rev(), slice);
        };

        assert_from_slice_roundtrip(&[]);
        assert_from_slice_roundtrip(&['a']);
        assert_from_slice_roundtrip(&['a', 'b']);
        assert_from_slice_roundtrip(&['a', 'b', 'c']);
    }

    #[test]
    fn test_from_vec() {
        let assert_from_vec_roundtrip = |vec: Vec<_>| {
            let clist = CList::from_vec(vec.clone());
            assert_eq!(clist.collect(), vec) ;
            assert_eq!(clist.collect_rev(), vec);
        };

        assert_from_vec_roundtrip(vec![]);
        assert_from_vec_roundtrip(vec!['a']);
        assert_from_vec_roundtrip(vec!['a', 'b']);
        assert_from_vec_roundtrip(vec!['a', 'b', 'c']);
    }


    #[test]
    fn test_insert_ref() {
        let mut clist = CList::with_capacity(5);
        let a_addr = clist.insert_ref('a', ROOT_ADDR, true);
        let c_addr = clist.insert_ref('c', a_addr, true);
        let _b_addr = clist.insert_ref('b', c_addr, false);
        assert_eq!(clist.collect(), vec!['a', 'b', 'c']);
        assert_eq!(clist.collect_rev(), vec!['a', 'b', 'c']);

        let _x_addr = clist.insert_ref('x', ROOT_ADDR, true);
        assert_eq!(clist.collect(), vec!['x', 'a', 'b', 'c']);
        assert_eq!(clist.collect_rev(), vec!['x', 'a', 'b', 'c']);

        let _y_addr = clist.insert_ref('y', ROOT_ADDR, false);
        assert_eq!(clist.collect(), vec!['x', 'a', 'b', 'c', 'y']);
        assert_eq!(clist.collect_rev(), vec!['x', 'a', 'b', 'c', 'y']);
    }

    #[test]
    fn test_len() {
        let mut clist = CList::with_capacity(2);
        assert_eq!(clist.len(), 0);

        let a_addr = clist.insert_ref('a', ROOT_ADDR, true);
        assert_eq!(clist.len(), 1);

        let b_addr = clist.insert_ref('b', a_addr, true);
        assert_eq!(clist.len(), 2);

        clist.remove(a_addr);
        assert_eq!(clist.len(), 1);

        clist.remove(b_addr);
        assert_eq!(clist.len(), 0);
    }

    #[test]
    fn test_is_empty() {
       let mut clist = CList::with_capacity(2);
       assert_eq!(clist.is_empty(), true);

       let a_addr = clist.insert_ref('a', ROOT_ADDR, true);
       assert_eq!(clist.is_empty(), false);

       clist.remove(a_addr);
       assert_eq!(clist.is_empty(), true);

    }

    #[should_panic]
    #[test]
    fn test_assert_not_free(){
        let mut clist = CList::with_capacity(1);
        let a_addr = clist.insert_ref('a', ROOT_ADDR, true);
        clist.remove(a_addr);
        clist.assert_not_free(&a_addr);
    }

    #[test]
    fn test_get() {
        let mut clist = CList::with_capacity(1);
        let a_addr = clist.insert_ref('a', ROOT_ADDR, true);
        assert_eq!(*clist.get(a_addr), 'a');
        assert_ne!(*clist.get(a_addr), 'b');
    }

    #[test]
    fn test_get_mut() {
        let mut clist = CList::with_capacity(1);
        let _a_addr = clist.insert_ref('a', ROOT_ADDR, true);
        let b_addr = clist.insert_ref('b', ROOT_ADDR, true);
        
        assert_eq!(*clist.get(b_addr), 'b');
        assert_ne!(*clist.get(b_addr), 'a');
    }

    #[test]
    fn test_iter() {
        let mut clist = CList::with_capacity(8);
        let c_addr = clist.insert_iter_right(ROOT_ADDR, "abc".chars());
        let _q_addr = clist.insert_ref('q', c_addr, false);

        let x_addr = clist.insert_iter_left(ROOT_ADDR, "xyz".chars());
        let _p_addr = clist.insert_ref('p', x_addr, true);

        assert_eq!(
            clist.collect(),
            vec!['a', 'b', 'q', 'c', 'x', 'p', 'y', 'z']
        );
        assert_eq!(
            clist.collect_rev(),
            vec!['a', 'b', 'q', 'c', 'x', 'p', 'y', 'z']
        );
    }

    #[test]
    fn test_remove() {
        let mut clist = CList::with_capacity(4);
        let a_addr = clist.insert_ref('a', ROOT_ADDR, true);
        let b_addr = clist.insert_ref('b', a_addr, true);
        let c_addr = clist.insert_ref('c', b_addr, true);
        let d_addr = clist.insert_ref('d', c_addr, true);

        assert_eq!(clist.remove(b_addr), c_addr);
        assert_eq!(clist.collect(), vec!['a', 'c', 'd']);
        assert_eq!(clist.collect_rev(), vec!['a', 'c', 'd']);

        assert_eq!(clist.remove(d_addr), ROOT_ADDR);
        assert_eq!(clist.collect(), vec!['a', 'c']);
        assert_eq!(clist.collect_rev(), vec!['a', 'c']);

        assert_eq!(clist.remove(a_addr), c_addr);
        assert_eq!(clist.collect(), vec!['c']);
        assert_eq!(clist.collect_rev(), vec!['c']);

        assert_eq!(clist.remove(c_addr), ROOT_ADDR);
        assert_eq!(clist.collect(), vec![]);
        assert_eq!(clist.collect_rev(), vec![]);
    }

    #[test]
    fn test_remove_n() {
        let mut clist = CList::with_capacity(5);
        let a_addr = clist.insert_ref('a', ROOT_ADDR, true);
        let b_addr = clist.insert_ref('b', a_addr, true);
        let _e_addr = clist.insert_iter_right(b_addr, "cde".chars());

        clist.remove_n(b_addr, 3);
        assert_eq!(clist.collect(), vec!['a', 'e']);
        assert_eq!(clist.collect_rev(), vec!['a', 'e']);
        assert_eq!(clist.len(), 2);

        clist.remove_n(a_addr, 2);
        assert_eq!(clist.collect(), vec![]);
        assert_eq!(clist.collect_rev(), vec![]);
        assert_eq!(clist.len(), 0);
    }

    #[test]
    fn test_remove_n_plus() {
        let mut clist = CList::with_capacity(5);
        let a_addr = clist.insert_ref('a', ROOT_ADDR, true);
        let b_addr = clist.insert_ref('b', a_addr, true);
        let _e_addr = clist.insert_iter_right(b_addr, "cde".chars());

        clist.remove_n(b_addr, 10);
        assert_eq!(clist.collect(), vec!['a']);
        assert_eq!(clist.collect_rev(), vec!['a']);
        assert_eq!(clist.len(), 1);
    }

    #[test]
    fn test_collect_n() {
        let clist = CList::from_vec("abcdefg".chars().collect());
        assert_eq!(clist.collect_n(0), vec![]);
        assert_eq!(clist.collect_n(3), vec!['a', 'b', 'c']);
        assert_eq!(clist.collect_n(7), vec!['a', 'b', 'c', 'd', 'e', 'f', 'g']);
        assert_eq!(clist.collect_n(10), vec!['a', 'b', 'c', 'd', 'e', 'f', 'g']);
        assert_eq!(clist.collect_n_from_pos(3, 1), vec!['b', 'c', 'd']);
        assert_eq!(clist.collect_n_from_pos(3, 0), vec!['a', 'b', 'c']);
        assert_eq!(clist.collect_n_from_pos(3, 5), vec!['f', 'g']);
        assert_eq!(clist.collect_n_from_pos(3, 6), vec!['g']);
        assert_eq!(clist.collect_n_from_pos(3, 7), vec![]);
    }

    #[test]
    #[should_panic]
    fn test_invalid_ref() {
        let mut clist = CList::with_capacity(2);
        let a_addr = clist.insert_ref('a', ROOT_ADDR, true);
        clist.remove(a_addr);
        let _b_addr = clist.insert_ref('b', a_addr, true);
    }

    #[test]
    fn test_free_reuse() {
        let mut clist = CList::with_capacity(2);
        let a_addr = clist.insert_ref('a', ROOT_ADDR, true);
        let b_addr = clist.insert_ref('b', a_addr, true);
        clist.remove(b_addr);
        clist.remove(a_addr);

        let c_addr = clist.insert_ref('c', ROOT_ADDR, true);
        let d_addr = clist.insert_ref('d', c_addr, true);
        assert_eq!(c_addr, a_addr);
        assert_eq!(d_addr, b_addr);
    }
}
