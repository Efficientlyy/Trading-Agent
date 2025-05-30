fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/market_data.proto");
    tonic_build::compile_protos("proto/market_data.proto")?;
    Ok(())
}
