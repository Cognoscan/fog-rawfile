use fog_pack::{
    document::Document,
    validator::*,
    schema::{Schema, SchemaBuilder, Compress},
};

use lazy_static::lazy_static;

lazy_static! {
    pub static ref DOC_SCHEMA_DATA: Document = doc_schema_data();
    pub static ref DOC_SCHEMA_LEVEL1: Document = doc_schema_level1();
    pub static ref DOC_SCHEMA_LEVEL2: Document = doc_schema_level2();
    pub static ref DOC_SCHEMA_LEVEL3: Document = doc_schema_level3();
    pub static ref SCHEMA_DATA: Schema = Schema::from_doc(&DOC_SCHEMA_DATA).unwrap();
    pub static ref SCHEMA_LEVEL1: Schema = Schema::from_doc(&DOC_SCHEMA_LEVEL1).unwrap();
    pub static ref SCHEMA_LEVEL2: Schema = Schema::from_doc(&DOC_SCHEMA_LEVEL2).unwrap();
    pub static ref SCHEMA_LEVEL3: Schema = Schema::from_doc(&DOC_SCHEMA_LEVEL3).unwrap();
}

fn doc_schema_data() -> Document {
    SchemaBuilder::new(BinValidator::new().build())
        .description("fog-rawfile: Raw binary data")
        .build()
        .unwrap()
}

fn doc_schema_level1() -> Document {
    SchemaBuilder::new(
        ArrayValidator::new()
            .comment("List of raw binary documents that form the file")
            .items(
                ArrayValidator::new()
                    .comment("Length of each binary document in bytes, followed by the document hash")
                    .min_len(2)
                    .max_len(2)
                    .prefix_add(IntValidator::new().min(1u64).build())
                    .prefix_add(HashValidator::new().schema_add(DOC_SCHEMA_DATA.hash()).build())
                    .build()
            ).build()
    )
        .description("fog-rawfile: Level 1 list of binary data")
        .doc_compress(Compress::None)
        .build()
        .unwrap()
}

fn doc_schema_level2() -> Document {
    SchemaBuilder::new(
        ArrayValidator::new()
            .comment("2nd-level list of binary data that forms the file")
            .items(
                ArrayValidator::new()
                    .comment("Combined length of data referenced by document, followed by document hash")
                    .min_len(2)
                    .max_len(2)
                    .prefix_add(IntValidator::new().min(1u64).build())
                    .prefix_add(HashValidator::new().schema_add(DOC_SCHEMA_LEVEL1.hash()).build())
                    .build()
            ).build()
    )
        .description("fog-rawfile: Level 2 list of binary data")
        .doc_compress(Compress::None)
        .build()
        .unwrap()
}

fn doc_schema_level3() -> Document {
    SchemaBuilder::new(
        ArrayValidator::new()
            .comment("3rd-level list of binary data that forms the file")
            .items(
                ArrayValidator::new()
                    .comment("Combined length of data referenced by document, followed by document hash")
                    .min_len(2)
                    .max_len(2)
                    .prefix_add(IntValidator::new().min(1u64).build())
                    .prefix_add(HashValidator::new().schema_add(DOC_SCHEMA_LEVEL2.hash()).build())
                    .build()
            ).build()
    )
        .description("fog-rawfile: Level 3 list of binary data")
        .doc_compress(Compress::None)
        .build()
        .unwrap()
}
