mod writing;

use anyhow::Result;
use crossterm::{
    event::Event,
    terminal::{disable_raw_mode, enable_raw_mode, size},
};
use std::sync::mpsc::{Receiver, Sender};

use crate::{
    clist::{CList, ROOT_ADDR},
    display::DisplayEvent,
};

pub(super) type Line = CList<char>;
type Text = CList<Line>;

const NULL_CHAR: char = '\0';

impl Line {
    fn init() -> (Self, usize) {
        let mut line = Line::with_capacity(1);
        let char_addr = line.insert_ref(NULL_CHAR, ROOT_ADDR, true);
        (line, char_addr)
    }
}

impl Text {
    fn init() -> (Self, usize, usize) {
        let mut text = Text::with_capacity(1);
        let (line, char_addr) = Line::init();
        let line_addr = text.insert_ref(line, ROOT_ADDR, true);
        (text, char_addr, line_addr)
    }
}

#[derive(PartialEq, Eq)]
enum EngineState {
    Writing,
    Quit,
}

pub struct Engine {
    state: EngineState,
    char_addr: usize,
    line_addr: usize,
    text: Text,
    window_width: u16,
    window_height: u16,
    cursor_pos_x: usize,
    cursor_pos_y: usize,
    _view_offset_x: usize,
    _view_offset_y: usize,
}

impl Engine {
    pub fn init() -> Result<Self> {
        let (text, char_addr, line_addr) = Text::init();
        let (window_width, window_height) = size()?;
        Ok(Self {
            state: EngineState::Writing,
            char_addr,
            line_addr,
            text,
            window_width,
            window_height,
            cursor_pos_x: 0,
            cursor_pos_y: 0,
            _view_offset_x: 0,
            _view_offset_y: 0,
        })
    }

    fn handle_input_event(
        &mut self,
        input_event: Event,
        display_event_sender: &Sender<DisplayEvent>,
    ) -> Result<Option<EngineState>> {
        match self.state {
            EngineState::Writing => {
                self.handle_input_event_writing(input_event, display_event_sender)
            }
            EngineState::Quit => unreachable!(),
        }
    }

    pub fn run(
        &mut self,
        input_event_receiver: &Receiver<Event>,
        display_event_sender: &Sender<DisplayEvent>,
    ) -> Result<()> {
        display_event_sender.send(DisplayEvent::EnterAlternateScreen)?;
        display_event_sender.send(DisplayEvent::MoveCursor(0, 0))?;
        enable_raw_mode()?;
        for input_event in input_event_receiver {
            if let Some(next_state) = self.handle_input_event(input_event, display_event_sender)? {
                if next_state == EngineState::Quit {
                    break;
                }
                self.state = next_state;
            }
            {
                // TODO(FIX): Remove this section.
                let text_height = self.window_height - 1;
                display_event_sender.send(DisplayEvent::PrintText(
                    self.text.collect_n(usize::from(text_height)),
                    self.window_width,
                    text_height,
                ))?;
                display_event_sender.send(DisplayEvent::MoveCursor(
                    self.cursor_pos_x.try_into().unwrap(),
                    self.cursor_pos_y.try_into().unwrap(),
                ))?;
            }
        }
        display_event_sender.send(DisplayEvent::LeaveAlternateScreen)?;
        disable_raw_mode()?;
        Ok(())
    }
}
