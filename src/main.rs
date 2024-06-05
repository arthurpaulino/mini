use anyhow::Result;
use crossterm::event::{read, Event};
use std::sync::mpsc::{self, Receiver, Sender};

use mini::{
    display::{run_display, DisplayEvent},
    engine::Engine,
};

fn run_input_reader_thread(input_event_sender: Sender<Event>) {
    std::thread::spawn(move || loop {
        let event = read().expect("Failed to read event.");
        input_event_sender
            .send(event)
            .expect("Failed to send event.");
    });
}

fn run_display_thread(display_event_receiver: Receiver<DisplayEvent>) {
    std::thread::spawn(move || run_display(display_event_receiver));
}

fn main() -> Result<()> {
    let (input_event_sender, input_event_receiver) = mpsc::channel();
    let (display_event_sender, display_event_receiver) = mpsc::channel();
    run_input_reader_thread(input_event_sender);
    run_display_thread(display_event_receiver);
    Engine::init()?.run(&input_event_receiver, &display_event_sender)
}
