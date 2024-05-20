use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyModifiers};
use std::sync::mpsc::Sender;

use crate::{clist::ROOT_ADDR, display::DisplayEvent};

use super::{Engine, EngineState, Line, NULL_CHAR};

fn normalize_char(c: &char, modifiers: &KeyModifiers) -> char {
    if modifiers == &KeyModifiers::SHIFT {
        c.to_ascii_uppercase()
    } else {
        *c
    }
}

impl Engine {
    fn move_left(&mut self) -> bool {
        let curr_line = self.text.get(self.line_addr);
        let prev_char_addr = curr_line.prev(self.char_addr);
        if prev_char_addr == ROOT_ADDR {
            // At the beginning of a line.
            let prev_line_addr = self.text.prev(self.line_addr);
            if prev_line_addr == ROOT_ADDR {
                // At the beginning of the text. Do nothing.
                return false;
            }
            let prev_line = self.text.get(prev_line_addr);
            self.char_addr = prev_line.last_addr();
            self.line_addr = prev_line_addr;
            self.cursor_pos_x = prev_line.len() - 1;
            self.cursor_pos_y -= 1;
            return true;
        }
        self.char_addr = prev_char_addr;
        self.cursor_pos_x -= 1;
        true
    }

    fn delete(&mut self) {
        let curr_line = self.text.get(self.line_addr);
        let null_char_addr = curr_line.last_addr();
        if self.char_addr == null_char_addr {
            // At the end of a line.
            let next_line_addr = self.text.next(self.line_addr);
            if next_line_addr == ROOT_ADDR {
                // End of text. Do nothing.
                return;
            }
            let next_line = self.text.get(next_line_addr);
            if curr_line.len() < next_line.len() {
                // Prepend the next line with the current line and then remove
                // the current line.
                let curr_line = self.text.get_mut(self.line_addr);
                curr_line.remove(null_char_addr);
                let curr_line_chars = curr_line.collect();
                let next_line = self.text.get_mut(next_line_addr);
                self.char_addr = next_line.first_addr();
                next_line.insert_iter_left(self.char_addr, curr_line_chars);
                self.text.remove(self.line_addr);
                self.line_addr = next_line_addr;
            } else {
                // Append the next line to the current line and then remove the
                // next line.
                let next_line_chars = next_line.collect();
                let curr_line = self.text.get_mut(self.line_addr);
                let null_char_prev_addr = curr_line.prev(null_char_addr);
                curr_line.remove(null_char_addr);
                curr_line.insert_iter_right(null_char_prev_addr, next_line_chars);
                self.char_addr = curr_line.next(null_char_prev_addr);
                self.text.remove(next_line_addr);
            }
            return;
        }
        let curr_line = self.text.get_mut(self.line_addr);
        self.char_addr = curr_line.remove(self.char_addr);
    }

    pub(super) fn handle_input_event_writing(
        &mut self,
        input_event: Event,
        _display_event_sender: &Sender<DisplayEvent>,
    ) -> Result<Option<EngineState>> {
        match input_event {
            Event::Resize(width, height) => {
                self.window_width = width;
                self.window_height = height;
                Ok(None)
            }
            Event::Key(key_event) => match &key_event.code {
                KeyCode::Esc => Ok(Some(EngineState::Quit)),
                KeyCode::Char(c) => {
                    let line = self.text.get_mut(self.line_addr);
                    let chr = normalize_char(c, &key_event.modifiers);
                    line.insert_ref(chr, self.char_addr, false);
                    self.cursor_pos_x += 1;
                    Ok(None)
                }
                KeyCode::Left => {
                    self.move_left();
                    Ok(None)
                }
                KeyCode::Right => {
                    let curr_line = self.text.get(self.line_addr);
                    if self.char_addr == curr_line.last_addr() {
                        // At the end of a line.
                        let next_line_addr = self.text.next(self.line_addr);
                        if next_line_addr == ROOT_ADDR {
                            // End of text. Do nothing.
                            return Ok(None);
                        }
                        self.char_addr = self.text.get(next_line_addr).first_addr();
                        self.line_addr = next_line_addr;
                        self.cursor_pos_x = 0;
                        self.cursor_pos_y += 1;
                        return Ok(None);
                    }
                    self.char_addr = curr_line.next(self.char_addr);
                    self.cursor_pos_x += 1;
                    Ok(None)
                }
                KeyCode::Enter => {
                    let curr_line = self.text.get_mut(self.line_addr);
                    let rest_capacity = curr_line.len() - self.cursor_pos_x;
                    let rest = curr_line.collect_from_addr(self.char_addr, Some(rest_capacity));
                    // TODO(OPTIMIZE): If there aren't too many chars to remove,
                    // use `remove_n`. Otherwise, create a new line with the remaining
                    // chars, insert it and remove the old line. Deciding whether
                    // there are too many chars to remove or not depends on benchmark
                    // numbers. Suppose there are N chars in a line and it's being
                    // broken after x chars. Is it faster to collect x elements
                    // and create a line `from_vec` with them or to remove N - x
                    // elements?
                    curr_line.remove_n(self.char_addr, curr_line.len());
                    curr_line.insert_ref(NULL_CHAR, ROOT_ADDR, false);
                    self.line_addr =
                        self.text
                            .insert_ref(Line::from_vec(rest), self.line_addr, true);
                    self.char_addr = self.text.get(self.line_addr).first_addr();
                    self.cursor_pos_x = 0;
                    self.cursor_pos_y += 1;
                    Ok(None)
                }
                KeyCode::Delete => {
                    self.delete();
                    Ok(None)
                }
                KeyCode::Backspace => {
                    if self.move_left() {
                        self.delete()
                    }
                    Ok(None)
                }
                _ => Ok(None),
            },
            _ => Ok(None),
        }
    }
}
