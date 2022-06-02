#![allow(dead_code)] // TODO: Remove when the API is complete

use std::{io::{Read, BufReader, self}, path::Path, fs::File};


mod schemas;
mod fixed;
mod config;
mod fastcdc;

pub use config::Config;
pub use fixed::FixedSize;

struct FastCdc<R: Read> {
    reader: R,
    data0: Vec<u8>,
    data1: Vec<u8>,
    compress: Option<Option<u8>>,
}

impl FastCdc<BufReader<File>> {

    fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let reader = BufReader::new(File::open(path)?);
        Ok(Self::from_reader(reader))
    }

    fn from_file_compress<P: AsRef<Path>>(path: P, compress: Option<u8>) -> io::Result<Self> {
        let reader = BufReader::new(File::open(path)?);
        Ok(Self::from_reader_compress(reader, compress))
    }
}

impl<R: Read> FastCdc<R> {
    fn from_reader(reader: R) -> Self {
        Self {
            reader,
            data0: Vec::new(),
            data1: Vec::new(),
            compress: None,
        }
    }

    fn from_reader_compress(reader: R, compress: Option<u8>) -> Self {
        Self {
            reader,
            data0: Vec::new(),
            data1: Vec::new(),
            compress: Some(compress),
        }
    }
}

impl<R: Read> Iterator for FastCdc<R> {
    type Item = fog_pack::document::Document;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
