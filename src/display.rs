use anyhow::Result;
use crossterm::{cursor, execute, queue, style, terminal};
use std::{
    io::{stdout, Write},
    sync::mpsc::Receiver,
};

use crate::engine::Line;

pub enum DisplayEvent {
    EnterAlternateScreen,
    LeaveAlternateScreen,
    WriteChar(char, u16, u16),
    PrintText(Vec<Line>, u16, u16),
    MoveCursor(u16, u16),
}

pub fn run_display(display_event_receiver: Receiver<DisplayEvent>) -> Result<()> {
    for display_event in display_event_receiver {
        match display_event {
            DisplayEvent::EnterAlternateScreen => {
                execute!(stdout(), terminal::EnterAlternateScreen)?;
            }
            DisplayEvent::LeaveAlternateScreen => {
                execute!(stdout(), terminal::LeaveAlternateScreen)?;
            }
            DisplayEvent::WriteChar(c, x, y) => {
                execute!(stdout(), cursor::MoveTo(x, y), style::Print(c))?;
            }
            DisplayEvent::PrintText(lines, width, height) => {
                let width = usize::from(width);
                let mut stdout = stdout();
                let num_lines = u16::try_from(lines.len())?;
                for (i, line) in lines.into_iter().enumerate() {
                    let mut line_str = String::with_capacity(width);
                    for c in line.collect_n(width) {
                        line_str.push(c);
                    }
                    let y = u16::try_from(i)?;
                    line_str.extend(vec![' '; width - line_str.len()]);
                    queue!(stdout, cursor::MoveTo(0, y), style::Print(line_str))?;
                }
                for i in 0..(height - num_lines) {
                    let mut empty_str = String::with_capacity(width);
                    empty_str.extend(vec![' '; width]);
                    queue!(
                        stdout,
                        cursor::MoveTo(0, num_lines + i),
                        style::Print(empty_str)
                    )?;
                }
                stdout.flush()?;
            }
            DisplayEvent::MoveCursor(x, y) => {
                execute!(stdout(), cursor::MoveTo(x, y))?;
            }
        }
    }
    Ok(())
}
