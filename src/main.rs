use std::fs::File;

mod models;

use models::HegselmannKrause;


fn main() -> std::io::Result<()> {
    let mut hk = HegselmannKrause::new(1000, 13);

    let outname: &'static str = "out.dat";
    let mut output = File::create(outname)?;
    let mut gp = File::create("out.gp")?;
    hk.write_gp(&mut gp, outname)?;

    for _ in 0..200 {
        hk.sweep();
        hk.write_state(&mut output)?;
    }
    Ok(())
}
