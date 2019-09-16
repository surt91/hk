use std::fs::File;

mod models;

use models::HegselmannKrause;
use models::HegselmannKrauseLorenz;

//
// fn main() -> std::io::Result<()> {
//     let mut hk = HegselmannKrause::new(1000, 13);
//
//     let outname: &'static str = "out_cell.dat";
//     let mut output = File::create(outname)?;
//     let mut gp = File::create("out_cell.gp")?;
//     hk.write_gp(&mut gp, outname)?;
//
//     for _ in 0..200 {
//         hk.sweep();
//         hk.write_state(&mut output)?;
//     }
//     Ok(())
// }

fn main() -> std::io::Result<()> {
    let mut hk = HegselmannKrauseLorenz::new(100, 3, 13);

    let outname: &'static str = "out_hkl.dat";
    let mut output = File::create(outname)?;
    let mut gp = File::create("out_hkl.gp")?;
    hk.write_gp(&mut gp, outname)?;

    for _ in 0..200 {
        hk.sweep();
        hk.write_state(&mut output)?;
    }
    Ok(())
}
