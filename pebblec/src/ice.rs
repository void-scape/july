use pebblec_parse::lex::io;
use std::panic::UnwindSafe;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::UNIX_EPOCH;

static BACKTRACE: LazyLock<Arc<Mutex<Option<String>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(None)));

pub fn reported_panic<Out>(
    capture: bool,
    note: &str,
    f: impl FnOnce() -> Out + UnwindSafe,
) -> Option<Out> {
    std::panic::set_hook(Box::new(panic_hook));

    if capture {
        // capture compiler output in case of panic
        std::io::set_output_capture(Some(Default::default()));

        let result = std::panic::catch_unwind(move || f());

        let capture = std::io::set_output_capture(None);
        let captured = Arc::try_unwrap(capture.unwrap())
            .unwrap()
            .into_inner()
            .unwrap();
        let output = String::from_utf8(captured).unwrap();

        print!("{output}");
        match result {
            Ok(result) => Some(result),
            Err(_) => {
                report_panic(&output, note);
                None
            }
        }
    } else {
        match std::panic::catch_unwind(move || f()) {
            Ok(result) => Some(result),
            Err(_) => {
                report_panic("", note);
                None
            }
        }
    }
}

fn retrieve_backtrace() -> String {
    format!(
        "\nBacktrace:\n{}",
        BACKTRACE
            .lock()
            .unwrap()
            .take()
            .unwrap_or("<Backtrace not found>".to_string())
    )
}

fn report_panic(stdout: &str, note: &str) {
    let ice_file = format!(
        "ICE-{}",
        std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );

    let note = format!("note: {}", note);
    let ice = format!("note: reported in `{ice_file}`");
    let notes = format!("\n{note}\n{ice}");
    println!("{notes}");

    let err = retrieve_backtrace();
    if let Err(io_err) = io::write(ice_file, format!("{stdout}{err}{notes}").as_bytes()) {
        println!("failed to report ICE: {io_err}");
    }
}

fn panic_hook(info: &std::panic::PanicHookInfo) {
    println!("{info}");
    *BACKTRACE.lock().unwrap() = Some(std::backtrace::Backtrace::force_capture().to_string());
}
