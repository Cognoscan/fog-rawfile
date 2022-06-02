
use std::{io::{Read, BufReader, self, ErrorKind}, path::Path, fs::File};
use crate::Config;

use fog_pack::{types::Hash, document::{Document, NewDocument}};
use serde_bytes::Bytes;

use crate::schemas;

/// A wrapper around a reader that converts it into one or more documents. When the conversion 
/// process is complete, the final document provided is always the top level document pointing to 
/// all sub-documents.
///
/// If the reader is empty, a single empty document will still be returned.
pub struct FixedSize<R: Read> {
    reader: R,
    // Config
    config: Config,
    // Document construction
    filled: usize,
    data: Vec<u8>,
    lvl1: Vec<(u64, Hash)>,
    lvl2: Vec<(u64, Hash)>,
    lvl3: Vec<(u64, Hash)>,
    doc_out1: Option<Document>,
    doc_out2: Option<Document>,
    doc_out3: Option<Document>,
    // State tracking
    nonempty: bool,
    lvl1_published: bool,
    lvl2_published: bool,
    done: bool,
}

impl FixedSize<BufReader<File>> {

    /// Start a new rawfile converter for a given file path.
    ///
    /// This is a convenience function around [`from_reader`][FixedSize::from_reader] that just 
    /// opens a buffered file reader.
    ///
    /// If additional configuration is desired, see [`Config`].
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let reader = BufReader::new(File::open(path)?);
        Ok(Self::from_reader(reader))
    }

}

impl<R: Read> FixedSize<R> {

    /// Start a new rawfile converter reading from the provided byte reader.
    ///
    /// If additional configuration is desired, see [`Config`].
    pub fn from_reader(reader: R) -> Self {
        Self::from_reader_config(reader, Config::default())
    }

    pub(crate) fn from_reader_config(reader: R, config: Config) -> Self {
        Self {
            reader,
            nonempty: false,
            filled: 0,
            done: false,
            lvl1: Vec::with_capacity(16384),
            lvl2: Vec::new(),
            lvl3: Vec::new(),
            doc_out1: None,
            doc_out2: None,
            doc_out3: None,
            lvl1_published: false,
            lvl2_published: false,
            data: vec![0u8; config.size],
            config,
        }
    }
}

impl<R: Read> Iterator for FixedSize<R> {
    type Item = io::Result<fog_pack::document::Document>;

    fn next(&mut self) -> Option<Self::Item> {
        // Return lists first, if we have any in storage
        if let Some(doc) = self.doc_out1.take() { return Some(Ok(doc)); }
        if let Some(doc) = self.doc_out2.take() { return Some(Ok(doc)); }
        if let Some(doc) = self.doc_out3.take() { return Some(Ok(doc)); }

        // When we're done, keep returning None
        if self.done { return None; }

        // Fetch data until we are done, have enough, or hit an error
        while self.filled < self.config.size {
            match self.reader.read(&mut self.data[self.filled..self.config.size]) {
                Ok(0) => { self.done = true; break },
                Ok(f) => self.filled += f,
                Err(e) if e.kind() == ErrorKind::Interrupted => (),
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            }
        }

        let ret = if self.filled > 0 || !self.nonempty {
            self.nonempty = true; // we got at least one byte, now we don't need to return an empty 
                                  // document.
            let doc = NewDocument::new(Bytes::new(&self.data[0..self.filled]), Some(schemas::SCHEMA_DATA.hash()))
                .unwrap();
            // Apply compression override to the data if configured to do so
            let doc = if let Some(setting) = self.config.compress {
                doc.compression(setting)
            }
            else {
                doc
            };
            self.lvl1.push((self.filled as u64, doc.hash()));
            Some(Ok(schemas::SCHEMA_DATA.validate_new_doc(doc).unwrap()))
        } else {
            None
        };

        self.filled = 0;

        if self.lvl1.len() >= self.config.lvl_size 
            || (self.done && !self.lvl1.is_empty() && self.lvl1_published) 
            || (self.done && self.lvl1.len() > 1)
        {
            self.lvl1_published = true;
            let len: u64 = self.lvl1.iter().map(|(l,_)| *l).sum();
            let doc = NewDocument::new(&self.lvl1, Some(schemas::SCHEMA_LEVEL1.hash())).unwrap();
            self.lvl1.clear();
            self.lvl2.push((len, doc.hash()));
            self.doc_out1 = Some(schemas::SCHEMA_LEVEL1.validate_new_doc(doc).unwrap());
        }

        if self.lvl2.len() >= self.config.lvl_size 
            || (self.done && !self.lvl2.is_empty() && self.lvl2_published) 
            || (self.done && self.lvl2.len() > 1)
        {
            self.lvl2_published = true;
            let len: u64 = self.lvl2.iter().map(|(l,_)| *l).sum();
            let doc = NewDocument::new(&self.lvl2, Some(schemas::SCHEMA_LEVEL2.hash())).unwrap();
            self.lvl2.clear();
            self.lvl3.push((len, doc.hash()));
            self.doc_out2 = Some(schemas::SCHEMA_LEVEL2.validate_new_doc(doc).unwrap());
        }

        if self.lvl3.len() >= self.config.lvl_size || (self.done && self.lvl3.len() > 1) {
            // If we hit this point and aren't done, we have to be. There can only one lvl3 
            // document, and if it's full then we must stop. Realistically I doubt anyone will ever 
            // hit this point.
            self.done = true;
            let doc = NewDocument::new(&self.lvl3, Some(schemas::SCHEMA_LEVEL3.hash())).unwrap();
            self.doc_out3 = Some(schemas::SCHEMA_LEVEL3.validate_new_doc(doc).unwrap());
        }

        if let Some(v) = ret {
            Some(v)
        }
        else {
            if let Some(doc) = self.doc_out1.take() { return Some(Ok(doc)); }
            if let Some(doc) = self.doc_out2.take() { return Some(Ok(doc)); }
            if let Some(doc) = self.doc_out3.take() { return Some(Ok(doc)); }
            None
        }
    }
}


#[cfg(test)]
mod tests {
    use std::{io::Read, cell::RefCell, rc::Rc};

    use fog_pack::types::Hash;
    use rand::SeedableRng;

    use crate::schemas;

    use super::*;

    fn schemas() {
        println!("schema data = {}", schemas::SCHEMA_DATA.hash());
        println!("schema lvl1 = {}", schemas::SCHEMA_LEVEL1.hash());
        println!("schema lvl2 = {}", schemas::SCHEMA_LEVEL2.hash());
        println!("schema lvl3 = {}", schemas::SCHEMA_LEVEL3.hash());
    }

    #[test]
    fn empty() {
        let data: &[u8] = &[];
        let mut r = FixedSize::from_reader(data);
        let doc = r.next().expect("Document").expect("No error, Valid document");
        assert!(r.next().is_none());

        let v: &[u8] = doc.deserialize().expect("byte vector");
        assert_eq!(v, data);
    }


    #[test]
    fn one_byte() {
        let data = vec![0u8; 1];
        let mut r = FixedSize::from_reader(data.as_slice());
        let doc = r.next().expect("Document").expect("No error, Valid document");
        assert!(r.next().is_none());

        let v: &[u8] = doc.deserialize().expect("byte vector");
        assert_eq!(v, data.as_slice());

    }

    #[test]
    fn size_bytes() {
        let size = 4096usize;
        let data: Vec<u8> = (0..size).map(|x| x.to_le_bytes()[0]).collect();
        let mut r = Config::default().block_size(size).build_fixed_reader(data.as_slice());
        //let mut r = FixedSize::from_reader(data.as_slice());
        let doc = r.next().expect("Document").expect("No error, Valid document");
        assert!(r.next().is_none());

        let v: &[u8] = doc.deserialize().expect("byte vector");
        assert_eq!(v, data.as_slice());
    }

    const MIN_SIZE: usize = 4096;
    const MIN_LVL_SIZE: usize = 64;

    #[test]
    fn size_plus_1() {
        verify_sequence(MIN_SIZE+1);
    }

    #[test]
    fn two_size() {
        verify_sequence(MIN_SIZE*2);
    }

    #[test]
    fn lvl1_full() {
        verify_sequence(MIN_SIZE*MIN_LVL_SIZE);
    }

    #[test]
    fn lvl2() {
        verify_sequence(MIN_SIZE*MIN_LVL_SIZE+1);
    }

    #[test]
    fn lvl2_two() {
        verify_sequence(MIN_SIZE*MIN_LVL_SIZE*2);
    }

    #[test]
    fn lvl2_full() {
        verify_sequence(MIN_SIZE*MIN_LVL_SIZE*MIN_LVL_SIZE);
    }

    #[test]
    fn lvl3() {
        verify_sequence(MIN_SIZE*MIN_LVL_SIZE*MIN_LVL_SIZE+1);
    }

    fn verify_sequence(len: usize) {
        let (fake_file, recv) = FakeBigFile::new(len);
        let mut chunker = Config::default()
            .block_size(MIN_SIZE)
            .level_size(MIN_LVL_SIZE)
            .build_fixed_reader(fake_file);
        let final_len = len % MIN_SIZE;

        let hash_data: Hash = schemas::SCHEMA_DATA.hash().clone();
        let hash_lvl1: Hash = schemas::SCHEMA_LEVEL1.hash().clone();
        let hash_lvl2: Hash = schemas::SCHEMA_LEVEL2.hash().clone();
        let hash_lvl3: Hash = schemas::SCHEMA_LEVEL3.hash().clone();

        #[derive(PartialEq,Eq,Clone,Copy,Debug)]
        enum State {
            Normal,
            End1,
            End2,
            End3,
            Done
        }

        let mut state = State::Normal;

        let mut lvl1 = Vec::new();
        let mut lvl2 = Vec::new();
        let mut lvl3 = Vec::new();
        let mut lvl1_got = false;
        let mut lvl2_got = false;
        let mut lvl3_got = false;

        for doc in chunker.by_ref() {
            let doc = doc.expect("Should never get an error from this reader");
            match doc.schema_hash() {
                Some(h) if *h == hash_data => {
                    // Check that this is what we should've received next
                    assert!(lvl1.len() < MIN_LVL_SIZE);
                    assert!(lvl2.len() < MIN_LVL_SIZE);
                    assert!(lvl3.len() < MIN_LVL_SIZE);
                    if state != State::Normal { panic!("Got 2nd data doc after we should've been done"); }
                    if recv.exhausted() {
                        state = if lvl1.is_empty() && !lvl1_got { State::Done } else { State::End1 };
                    }
                             
                    let v: &[u8] = doc.deserialize().expect("byte vector");
                    recv.check_chunk(v).unwrap();
                    if state != State::Normal {
                        if final_len == 0 {
                            assert_eq!(v.len(), MIN_SIZE);
                        }
                        else {
                            assert_eq!(v.len(), final_len);
                        }
                    }

                    lvl1.push((v.len() as u64, doc.hash().clone()));
                },
                Some(h) if *h == hash_lvl1 => {
                    lvl1_got = true;
                    // Check that this is what we should've received next
                    assert!(lvl2.len() < MIN_LVL_SIZE);
                    assert!(lvl3.len() < MIN_LVL_SIZE);
                    assert!((lvl1.len() == MIN_LVL_SIZE) || state == State::End1,
                        "lvl1.len() = {}, state = {:?}", lvl1.len(), state);
                    if state == State::End1 {
                        state = if lvl2.is_empty() && !lvl2_got { State::Done } else { State::End2 };
                    }

                    let v: Vec<(u64, Hash)> = doc.deserialize().expect("links to other docs");
                    assert!(v == lvl1, "lvl1 doc note equal to expected, doc.len = {}, expected = {}",
                        v.len(), lvl1.len());

                    let len: u64 = v.iter().map(|(l,_)| l).sum();
                    lvl1.clear();
                    lvl2.push((len, doc.hash().clone()));
                },
                Some(h) if *h == hash_lvl2 => {
                    lvl2_got = true;
                    // Check that this is what we should've received next
                    assert!(lvl1.len() < MIN_LVL_SIZE);
                    assert!(lvl3.len() < MIN_LVL_SIZE);
                    assert!((lvl2.len() == MIN_LVL_SIZE) || state == State::End2);
                    if state == State::End2 {
                        state = if lvl3.is_empty() { State::Done } else { State::End3 };
                    }

                    let v: Vec<(u64, Hash)> = doc.deserialize().expect("links to other docs");
                    assert_eq!(v, lvl2);

                    let len: u64 = v.iter().map(|(l,_)| l).sum();
                    lvl2.clear();
                    lvl3.push((len, doc.hash().clone()));
                },
                Some(h) if *h == hash_lvl3 => {
                    if lvl3_got { panic!("Got more than one LVL3 document"); }
                    lvl3_got = true;
                    // Check that this is what we should've received next
                    assert!(lvl1.is_empty());
                    assert!(lvl2.is_empty());
                    assert!((lvl3.len() == MIN_LVL_SIZE) || state == State::End3);
                    state = State::Done;

                    let v: Vec<(u64, Hash)> = doc.deserialize().expect("links to other docs");
                    assert_eq!(v, lvl3);

                    let lvl3_len: u64 = v.iter().map(|(l,_)| l).sum();
                    assert_eq!(lvl3_len, len as u64);
                    lvl3.clear();
                },
                Some(s) => panic!("Got a document with unknown schema: {}", s),
                None => panic!("Got a document without a schema"),
            }
        }

        assert!(chunker.next().is_none(), "Repeated polling of the iterator should stay at None");

        assert_eq!(state, State::Done); // Verify we reached the end in an expected way
    }


    struct FakeBigFile {
        rand: rand::rngs::SmallRng,
        data_recv: DataRecv,
        cnt: usize,
        len: usize,
    }

    #[derive(Clone)]
    struct DataRecv(Rc<RefCell<DataRecvInner>>);

    impl DataRecv {
        fn exhausted(&self) -> bool {
            self.0.borrow().exhausted()
        }

        fn check_chunk(&self, chunk: &[u8]) -> Result<(),String> {
            self.0.borrow_mut().check_chunk(chunk)
        }
    }


    struct DataRecvInner {
        cnt: usize,
        len: usize,
        data_current: Vec<u8>,
        data_next: Vec<u8>,
    }

    impl DataRecvInner {
        fn exhausted(&self) -> bool {
            self.len == self.cnt
        }

        // Confirm the chunk matches the bytes most recently written. Should be called with data in 
        // order
        fn check_chunk(&mut self, chunk: &[u8]) -> Result<(),String> {
            let len = chunk.len();
            if self.data_current.len() < len {
                return Err(
                    format!("Got {} bytes in chunk but only have {} bytes in buf", len, self.data_current.len())
                );
            }
            if &self.data_current[..len] == chunk {
                self.data_next.clear();
                self.data_next.extend_from_slice(&self.data_current[len..]);
                std::mem::swap(&mut self.data_next, &mut self.data_current);
                Ok(())
            }
            else {
                self.data_next.clear();
                self.data_next.extend_from_slice(&self.data_current[len..]);
                std::mem::swap(&mut self.data_next, &mut self.data_current);
                Err(String::from("Data in buffer and actual data don't match up"))
            }
        }
    }

    impl FakeBigFile {
        fn new(len: usize) -> (Self, DataRecv) {
            let data_recv = DataRecv(Rc::new(RefCell::new(DataRecvInner {
                len,
                cnt: 0,
                data_current: Vec::new(),
                data_next: Vec::new(),
            })));
            (Self {
                data_recv: data_recv.clone(),
                rand: rand::rngs::SmallRng::from_entropy(),
                cnt: 0,
                len,
            },
            data_recv,
            )
        }


    }

    impl Read for FakeBigFile {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            use rand::RngCore;
            if self.cnt >= self.len {
                Ok(0)
            }
            else if (self.cnt + buf.len()) > self.len {
                let out_len = self.len - self.cnt;
                self.rand.fill_bytes(&mut buf[..out_len]);
                let mut recv = self.data_recv.0.borrow_mut();
                recv.data_current.extend_from_slice(&buf[..out_len]);
                recv.cnt = self.len;
                self.cnt = self.len;
                Ok(out_len)
            }
            else {
                self.rand.fill_bytes(buf);
                let mut recv = self.data_recv.0.borrow_mut();
                recv.data_current.extend_from_slice(buf);
                recv.cnt += buf.len();
                self.cnt += buf.len();
                Ok(buf.len())
            }
        }
    }

    fn arb_chunk_test() {

    }
}
