use std::{io::{Read, BufReader, self}, path::Path, fs::File};
use crate::fixed::FixedSize;
use crate::fastcdc::FastCdc;

/// File stream configuration options.
///
/// This lets you set whether the data chunks should be compressed or not, what the size of each 
/// block should be, and how many hashes will be packed together at each level of the file's tree 
/// of data. 
///
/// By default:
///
/// - Standard compression settings are used (currently zstd at level 3)
/// - 64 kB chunks are used
/// - 16384 (2^14) hashes are packed in at each level of the tree
///
///
/// # Why Tune?
///
/// You probably shouldn't! Using the normal settings and the FastCdc chunker is great for most 
/// cases. But perhaps you know something special about your data that justifies additional 
/// configuration:
///
///
/// ## Chunking
///
/// The dynamic Content Defined Chunker is great for deduplication and working with files that 
/// cannot be parsed into fog-pack easily. It makes a great default. If, however, you know that the 
/// file format is block-oriented and inserts/deletes at arbitrary points within the file basically 
/// don't happen, then a fixed size chunker may make more sense. Certain on-disk database files and 
/// raw disk archives may fit this description, for example.
///
///
/// ## Compression
///
/// If your data is already optimally compressed, then adding generic compression is not helpful. 
/// This is true for most media files - audio/video/images/etc. as well as other compressed archive 
/// formats (eg. docx and other Office files). For these cases, overriding the compression saves 
/// time on encoding by not trying and giving up on compression.
///
///
/// ## Chunk sizes
///
/// For the dynamic Content Defined Chunker ([`FastCdc`]), you should not change 
/// this. Deduplication with CDC only works when all chunking is done with the same parameters, so 
/// this basically defeats the whole point. So while you *can* change this... don't.
///
/// For fixed size chunking, you hopefully know what size blocks are used by your file format (or 
/// whatever byte stream you're reading from). Pick that for optimal deduplication.
///
///
/// ## Number of hashes at each tree level
///
/// There's very few reasons to tweak this number. The tree documents can be updated less 
/// frequently and are more easily deduplicated when this number is smaller, but this should 
/// already be a small portion of your overall file size (<0.1%). Unless there's some unusual case, 
/// don't bother tweaking this.
///
#[derive(Clone,Debug)]
pub struct Config {
    pub(crate) compress: Option<Option<u8>>,
    pub(crate) size: usize,
    pub(crate) lvl_size: usize,
}

const DEFAULT_SIZE: usize = 32<<10;
const DEFAULT_LVL_SIZE: usize = 16<<10;

impl Default for Config {
    fn default() -> Self {
        Self {
            compress: None,
            size: DEFAULT_SIZE,
            lvl_size: DEFAULT_LVL_SIZE,
        }
    }
}

impl Config {

    /// Construct a Config object with default settings (identical to [`Config::default()`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set compression to either None (no compression) or a compression level understood by the 
    /// standard compression algorithm. Currently this is zstd, and any future algorithms will 
    /// attempt to provide the same scale of speed / compression level tradeoff at each setting.
    pub fn compress(&mut self, compress: Option<u8>) -> &mut Self {
        self.compress = Some(compress);
        self
    }

    /// Set the target size to use for each block. This should be between 4 kB and 512 kB -
    /// the configuration will automatically clamp it to that range if something outside of that is 
    /// provided.
    ///
    /// By default this is set to 32 kiB.
    pub fn block_size(&mut self, size: usize) -> &mut Self {
        self.size = size.clamp(4<<10, 512<<10);
        self
    }

    /// Set the fixed number of sub-unit hashes to use at each level of the file's tree of data. At 
    /// the lowest level (level 1) this is the number of block hashes, at level 2 it is the number 
    /// of hashes of level 1 documents, and at level 3 it is the number of hashes of level 2 
    /// documents. This should be between 64 and 20000 - configuration will automatically clamp it 
    /// to that range.
    ///
    /// By default this is set to 16384.
    pub fn level_size(&mut self, size: usize) -> &mut Self {
        self.lvl_size = size.clamp(64, 20000);
        self
    }

    /// Start a new rawfile converter for a given file path, using fixed chunk sizes.
    ///
    /// This is a convenience function around [`build_fixed_reader`][Config::build_fixed_reader] 
    /// that just opens a buffered file reader.
    pub fn build_fixed_file<P: AsRef<Path>>(&self, path: P) -> io::Result<FixedSize<BufReader<File>>> {
        let reader = BufReader::new(File::open(path)?);
        Ok(self.build_fixed_reader(reader))
    }

    /// Start a rawfile converter reading from the provided byte reader, using fixed chunk sizes.
    pub fn build_fixed_reader<R: Read>(&self, reader: R) -> FixedSize<R> {
        FixedSize::from_reader_config(reader, self.clone())
    }

    /// Start a new rawfile converter for a given file path, using a Content Defined Chunking 
    /// mechanism to assist with data deduplication.
    ///
    /// This is a convenience function around [`build_fixed_reader`][Config::build_fixed_reader] 
    /// that just opens a buffered file reader.
    pub fn build_file<P: AsRef<Path>>(&self, path: P) -> io::Result<FastCdc<BufReader<File>>> {
        let reader = BufReader::new(File::open(path)?);
        Ok(self.build_reader(reader))
    }

    /// Start a rawfile converter reading from the provided byte reader, using a Content Defined 
    /// Chunking mechanism to assist with data deduplication.
    pub fn build_reader<R: Read>(&self, reader: R) -> FastCdc<R> {
        //let mut config = self.clone();
        //config.size = DEFAULT_SIZE;
        FastCdc::from_reader_config(reader, self.clone())
    }

}
