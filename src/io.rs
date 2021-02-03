use std::fs::File;
use std::path::{PathBuf, Path};

use std::process::Command;


use rand::Rng;

pub struct Output {
    tmp_file: File,
    tmp_path: PathBuf,
    final_path: PathBuf,
}

impl Output {
    pub fn new(outname: &Path, extension: &str, tmp_path: &Path) -> std::io::Result<Output> {
        use rand::{thread_rng};
        use rand::distributions::Alphanumeric;
        let random: String = thread_rng()
            .sample_iter(&Alphanumeric)
            .take(10)
            .collect();

        let tmp_path = tmp_path.join(random).with_extension(extension);

        let final_path = outname.with_extension(extension);

        if let Some(dirs) = tmp_path.parent() {
            std::fs::create_dir_all(&dirs)?;
        }
        let tmp_file = File::create(&tmp_path)?;

        Ok(Output {
            tmp_file,
            tmp_path,
            final_path,
        })
    }

    pub fn file(&mut self) -> &mut File {
        &mut self.tmp_file
    }

    pub fn final_name(&self) -> PathBuf {
        let mut gz_ext = self.tmp_path.extension().unwrap().to_os_string();
        gz_ext.push(".gz");
        self.final_path.with_extension(&gz_ext)
    }

    fn zip(name: &Path) {
        Command::new("gzip")
            .arg(name.to_str().unwrap())
            .output()
            .expect("failed to zip output file");
    }

    pub fn finalize(self) -> std::io::Result<()> {
        let tmp_path = self.tmp_path;
        // flush and close temporary file
        self.tmp_file.sync_all()?;
        drop(self.tmp_file);

        // zip temporary file
        Output::zip(&tmp_path);
        let mut gz_ext = tmp_path.extension().unwrap().to_os_string();
        gz_ext.push(".gz");
        let tmp_path = tmp_path.with_extension(&gz_ext);
        let final_path = self.final_path.with_extension(&gz_ext);

        // move finished file to final location, (if they differ)
        if let Some(dirs) = final_path.parent() {
            std::fs::create_dir_all(&dirs)?;
        }
        if tmp_path != final_path {
            std::fs::rename(&tmp_path, &final_path).or_else(|_| {
                std::fs::copy(&tmp_path, &final_path).expect("could not move or copy the file");
                std::fs::remove_file(&tmp_path)
            })?;
        }

        Ok(())
    }
}
