import pygame
import pymunk
import pymunk.pygame_util
import random
import sys
import math
import colorsys
from moviepy import *
import pygame.midi
import numpy as np
from scipy.io.wavfile import write
from mido import MidiFile
from moviepy import *
import os
from scipy.io import wavfile
from datetime import datetime  # Import datetime for timestamp
import argparse  # Import argparse for command-line arguments
from common.marble import Marble  # Import the Marble class
from common.confetti import ConfettiManager  # Import ConfettiManager

# --- Configuration ---
FRAMERATE = 120
WIDTH, HEIGHT = 540, 960  # 9:16 aspect ratio
MARBLE_RADIUS = 10
NUM_MARBLES = 5
COLORS = [
    {"rgb": (255, 0, 0), "name": "Red"},
    {"rgb": (0, 255, 0), "name": "Green"},
    {"rgb": (31, 127, 255), "name": "Blue"},
    {"rgb": (255, 255, 0), "name": "Yellow"},
    {"rgb": (255, 165, 0), "name": "Orange"}
]
SIMULATION_TIME = 60  # seconds
CONFETTI_COUNT = 1000
SPARKLE_LIFETIME = 0.5
TRAIL_LENGTH = FRAMERATE / 3
PINWHEEL_TRAIL_LENGTH = FRAMERATE / 10

# --- Setup ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Marble Race")
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)
draw_options.shape_outline_color = (255,255,255,255)
draw_options.shape_static_color = (255,255,255,255)

space = pymunk.Space()
space.gravity = (0, 900)

confetti_manager = ConfettiManager(WIDTH, HEIGHT, COLORS, CONFETTI_COUNT)  # Initialize ConfettiManager

# --- Start and Finish ---
start_x, start_y = 85, 25
finish_area = pygame.Rect(WIDTH-50, HEIGHT - 200, 50, 50)
winner = None
confetti_particles = []
win_audio_data = None
win_audio_sample_rate = None

frame_counter = 0  # Initialize frame counter

# --- Create Marbles ---
marbles = []

def create_marbles():
    global marbles
    marbles = [Marble(i, start_x, start_y, space) for i in range(NUM_MARBLES)]  # Pass space explicitly

# --- Obstacle Class ---
class Obstacle:
    @staticmethod
    def add_static_segment(p1, p2):
        segment = pymunk.Segment(space.static_body, p1, p2, 5)
        segment.elasticity = 0.6
        segment.gradient_start = p1
        segment.gradient_end = p2
        space.add(segment)

    @staticmethod
    def add_bumper(x, y):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (x, y)
        shape = pymunk.Circle(body, 25)
        bumper_hits[shape] = 0
        shape.elasticity = 3.0
        shape.collision_type = 1  # Set collision type for bumper
        space.add(body, shape)

    @staticmethod
    def add_pinwheel(x, y, bottom_ramp=False, top_ramp=False):
        pinwheel_body = pymunk.Body(5, pymunk.moment_for_circle(5, 0, 40))
        pinwheel_body.position = (x, y)
        pinwheel_shapes = []
        for angle_deg in range(0, 360, 90):
            angle_rad = math.radians(angle_deg)
            dx = math.cos(angle_rad) * 40
            dy = math.sin(angle_rad) * 40
            shape = pymunk.Segment(pinwheel_body, (0, 0), (dx, dy), 4)
            shape.elasticity = 1.0
            shape.collision_type = 2  # Set collision type for pinwheel
            pinwheel_shapes.append(shape)
        space.add(pinwheel_body, *pinwheel_shapes)
        pivot = pymunk.PivotJoint(pinwheel_body, space.static_body, (x, y))
        pivot.collide_bodies = False
        space.add(pivot)
        motor_speed = (
            random.choice([-FRAMERATE/8, FRAMERATE/8]) if bottom_ramp else
            random.choice([-FRAMERATE/24, FRAMERATE/24]) if top_ramp else
            random.choice([-FRAMERATE/8, -FRAMERATE/12, -FRAMERATE/24, FRAMERATE/24, FRAMERATE/12, FRAMERATE/8])
        )
        motor = pymunk.SimpleMotor(space.static_body, pinwheel_body, motor_speed)
        space.add(motor)

    @staticmethod
    def add_obstacle(x, y, bottom_ramp=False, top_ramp=False):
        if random.random() < 0.5:
            Obstacle.add_bumper(x, y)
        else:
            Obstacle.add_pinwheel(x, y, bottom_ramp=bottom_ramp, top_ramp=top_ramp)

bumper_hits = {}

# --- MIDI Setup ---
midi_notes = []
current_note_index = 0
audio_recording = []
last_note_time = -0.125  # Initialize last note time to ensure the first note is played

def initialize_midi():
    pygame.midi.init()
    print("MIDI initialized.")

def close_midi():
    pygame.midi.quit()
    print("MIDI closed.")

def load_random_midi():
    global midi_notes
    midi_folder = os.path.join(os.path.dirname(__file__), "midi")
    midi_files = [f for f in os.listdir(midi_folder) if f.endswith(".mid")]
    if not midi_files:
        print("No MIDI files found in the 'midi' folder.")
        return
    selected_file = random.choice(midi_files)
    print(f"Selected MIDI file: {selected_file}")
    midi_path = os.path.join(midi_folder, selected_file)
    midi = MidiFile(midi_path)
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:  # Exclude rests
                midi_notes.append(msg)

def play_next_midi_note():
    global current_note_index, last_note_time
    current_time = frame_counter / FRAMERATE  # Calculate the current time in seconds
    if midi_notes and (current_time - last_note_time >= 0.125):  # Check time since last note
        note = midi_notes[current_note_index]
        print(f"Playing note: {note.note}, Velocity: {note.velocity}")
        audio_recording.append((frame_counter, note.note, note.velocity))  # Use frame_counter
        current_note_index = (current_note_index + 1) % len(midi_notes)  # Loop back to the start
        last_note_time = current_time  # Update the last note time

def play_win_sound():
    win_folder = os.path.join(os.path.dirname(__file__), "wav", "win")
    win_files = [f for f in os.listdir(win_folder) if f.endswith(".wav")]
    if not win_files:
        print("No win sound files found in the 'wav/win' folder.")
        return None, None

    selected_file = random.choice(win_files)
    print(f"Playing win sound: {selected_file}")
    win_path = os.path.join(win_folder, selected_file)
    win_sample_rate, win_data = wavfile.read(win_path)

    # Convert stereo to mono if necessary
    if len(win_data.shape) == 2:  # Check if stereo
        win_data = win_data.mean(axis=1)  # Convert to mono by averaging channels

    # Normalize and scale the win sound
    win_data = win_data.astype(np.float32) / np.max(np.abs(win_data))  # Normalize
    return win_data, win_sample_rate

def save_recording(winner_time):
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(os.path.dirname(__file__), "vid")
    os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists
    filename = os.path.join(output_folder, f"marble_race_{timestamp}.mp4")
    
    video_clip = ImageSequenceClip(recording_frames, fps=FRAMERATE)

    # Generate audio from recorded MIDI notes and win sound
    audio_file = "marble_race_audio.wav"
    sample_rate = 44100
    total_duration = len(recording_frames) / FRAMERATE  # Total duration of the video in seconds
    audio_data = np.zeros(int(total_duration * sample_rate), dtype=np.float32)  # Initialize audio array

    for frame_index, note, velocity in audio_recording:
        # Calculate the start time and sample index for the note
        start_time = frame_index / FRAMERATE
        start_sample = int(start_time * sample_rate)

        # Load the corresponding .wav file for the note
        wav_path = os.path.join(os.path.dirname(__file__), "wav/notes", f"{note}.wav")
        if not os.path.exists(wav_path):
            print(f"Warning: Missing .wav file for note {note}. Skipping.")
            continue

        wav_sample_rate, wav_data = wavfile.read(wav_path)
        if wav_sample_rate != sample_rate:
            print(f"Warning: Sample rate mismatch for {wav_path}. Resampling is required.")
            continue

        # Normalize and scale the note's audio data by velocity
        wav_data = wav_data.astype(np.float32) / np.max(np.abs(wav_data))  # Normalize
        wav_data *= (velocity / 127.0) * 0.65  # Scale by velocity and reduce volume to 65%

        # Add the note's waveform to the audio data, ensuring no overflow
        end_sample = start_sample + len(wav_data)
        audio_data[start_sample:end_sample] += wav_data[:len(audio_data[start_sample:end_sample])]

    # Add the win sound to the audio data
    if win_audio_data is not None and winner_time is not None:
        start_sample = int(winner_time * sample_rate)
        end_sample = start_sample + len(win_audio_data)
        audio_data[start_sample:end_sample] += win_audio_data[:len(audio_data[start_sample:end_sample])]

    # Normalize the final audio data to prevent clipping
    if np.max(np.abs(audio_data)) > 0:
        audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
        write(audio_file, sample_rate, audio_data)  # Save audio to WAV file
    else:
        print("No audio data generated. Skipping audio file creation.")
        audio_file = None

    # Combine video and audio
    if audio_file:
        audio_clip = AudioFileClip(audio_file)
        final_clip = video_clip.with_audio(audio_clip)
    else:
        final_clip = video_clip

    final_clip.write_videofile(filename, codec="libx264", audio_codec="aac")
    print(f"Recording saved as {filename}")

# --- Collision Handling ---
def handle_bumper_collision(arbiter, space, data):
    for shape in arbiter.shapes:
        if shape in bumper_hits:
            bumper_hits[shape] = FRAMERATE // 4
            play_next_midi_note()  # Play the next MIDI note on collision
    return True

def handle_pinwheel_collision(arbiter, space, data):
    play_next_midi_note()  # Play the next MIDI note on collision
    return True

def setup_collision_handlers():
    handler = space.add_collision_handler(0, 1)
    handler.post_solve = handle_bumper_collision

    pinwheel_handler = space.add_collision_handler(0, 2)  # Add handler for pinwheel collisions
    pinwheel_handler.post_solve = handle_pinwheel_collision

def generate_course():
    y = 200
    direction = 1  # Start with slanting right
    num_ramps = 4
    vertical_spacing = 130

    for i in range(num_ramps):
        if direction == 1:
            x1, x2, bumper_x = 0, WIDTH - 100, WIDTH - 50
        else:
            x1, x2, bumper_x = WIDTH, 100, 50

        Obstacle.add_static_segment((x1, y - 10), (x2, y + 40))

        if i == 0:
            # Top ramp: always add a spinner
            Obstacle.add_obstacle(bumper_x, y + 65, top_ramp=True)
        elif i == num_ramps - 1:
            # Bottom ramp: always add a spinner
            Obstacle.add_obstacle(bumper_x, y + 65, bottom_ramp=True)
        else:
            # Other ramps: randomly add either a bumper or a spinner
            Obstacle.add_obstacle(bumper_x, y + 65)

        center_x = WIDTH // 2
        offset = -75 if bumper_x > center_x else 75
        alt_x, alt_y = bumper_x + offset, y - 35
        Obstacle.add_obstacle(alt_x, alt_y)

        y += vertical_spacing
        direction *= -1

    # Bottom ramp
    Obstacle.add_static_segment((0, HEIGHT - 200), (WIDTH, HEIGHT - 150))
    Obstacle.add_static_segment((0, 0), (0, HEIGHT))  # Left wall
    Obstacle.add_static_segment((WIDTH, 0), (WIDTH, HEIGHT))  # Right wall
    Obstacle.add_static_segment((0, 0), (WIDTH, 0))  # Top wall

def check_for_winner():
    global winner, win_audio_data, win_audio_sample_rate
    for marble in marbles:
        if finish_area.collidepoint(marble.body.position.x, marble.body.position.y) and winner is None:
            winner = {"rgb": marble.color, "name": marble.name}
            confetti_manager.spawn_confetti()  # Use ConfettiManager to spawn confetti
            win_audio_data, win_audio_sample_rate = play_win_sound()  # Play and store win sound
            return True
    return False

# --- Elimination Effect ---
eliminated_effects = []

def spawn_elimination_burst(x, y, color):
    for _ in range(60):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(100, 500)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        eliminated_effects.append({
            'pos': [x, y],
            'vel': [vx, vy],
            'color': color,
            'life': 1.0
        })

def update_elimination_effects(dt):
    for p in eliminated_effects:
        p['pos'][0] += p['vel'][0] * dt
        p['pos'][1] += p['vel'][1] * dt
        p['life'] -= dt
    eliminated_effects[:] = [p for p in eliminated_effects if p['life'] > 0]

def draw_elimination_effects():
    for p in eliminated_effects:
        alpha = int(255 * p['life'])
        surf = pygame.Surface((10, 10), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*p['color'], alpha), (3, 3), 5)
        screen.blit(surf, (p['pos'][0] - 3, p['pos'][1] - 3))

# --- Draw Bumpers ---
def draw_bumpers():
    t = frame_counter / FRAMERATE  # Time in seconds for animation
    for shape in space.shapes:
        if isinstance(shape, pymunk.Circle) and shape in bumper_hits:
            pos = shape.body.position
            radius = int(shape.radius)
            gradient_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            for r in range(radius, 0, -1):
                color_shift = (math.sin(-t * 4 + r / radius * math.pi) + 1) * 64  # Oscillate color
                color = (128 + int(color_shift), int(color_shift), 255)  # Animated pink/purple gradient
                pygame.draw.circle(gradient_surface, (*color, 255), (radius, radius), r)
            screen.blit(gradient_surface, (int(pos.x - radius), int(pos.y - radius)))

            # Draw classic yellow lightning bolt at the center
            bolt_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            bolt_points = [
                (radius * 0.4, radius * 0.1),
                (radius * 0.6, radius * 0.1),
                (radius * 0.5, radius * 0.4),
                (radius * 0.7, radius * 0.4),
                (radius * 0.4, radius * 0.9),
                (radius * 0.5, radius * 0.6),
                (radius * 0.3, radius * 0.6),
                (radius * 0.4, radius * 0.3),
                (radius * 0.2, radius * 0.3),
                (radius * 0.3, radius * 0.1),
                (radius * 0.4, radius * 0.1)  # Close the shape
            ]
            scaled_points = [(x*1.2, y*1.2) for x, y in bolt_points]
            pygame.draw.polygon(bolt_surface, (0, 0, 0), scaled_points, width=2)
            pygame.draw.polygon(bolt_surface, (255, 255, 0), scaled_points)
            screen.blit(bolt_surface, (int(pos.x - 0.5*radius), int(pos.y - 0.5*radius)))

    for shape, timer in bumper_hits.items():
        if timer > 0:
            pos = shape.body.position
            t = timer / FRAMERATE
            pulse = 1 + 0.125 * (1 - math.cos(min(t * 8 * math.pi, math.pi)))
            radius = int(shape.radius * pulse)
            pygame.draw.circle(screen, color, (int(pos.x), int(pos.y)), radius, 2)
            bumper_hits[shape] -= 1

# --- Recording Setup ---
recording_frames = []

# --- Main Loop ---
def main():
    global marbles, winner, frame_counter, confetti_particles, eliminated_effects, recording_frames, midi_notes, current_note_index, last_note_time, audio_recording

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the marble race simulation.")
    parser.add_argument("runs", type=int, help="Number of simulation runs")
    args = parser.parse_args()

    for run in range(1, args.runs + 1):
        print(f"Starting simulation run {run}/{args.runs}")

        # Reset global variables
        marbles = []
        winner = None
        frame_counter = 0
        confetti_particles.clear()
        eliminated_effects.clear()
        recording_frames.clear()
        midi_notes = []
        current_note_index = 0
        last_note_time = -0.125
        audio_recording = []  # Clear audio recording

        # Clear Pygame and Pymunk space
        space.remove(*space.bodies, *space.shapes)

        initialize_midi()  # Initialize MIDI
        load_random_midi()  # Load a random MIDI file at the start
        create_marbles()
        generate_course()
        setup_collision_handlers()

        try:
            start_ticks = pygame.time.get_ticks()
            running = True
            winner_time = None  # Track when the winner is determined

            while running:
                clock.tick(FRAMERATE)
                frame_counter += 1  # Increment frame counter
                screen.fill((30, 30, 30))

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            running = False  # Stop the simulation if spacebar is pressed

                # Physics Step
                dt = 1 / FRAMERATE
                space.step(dt)

                # Draw finish area
                tile_size = 5
                for row in range(finish_area.height // tile_size):
                    for col in range(finish_area.width // tile_size):
                        color = (255, 255, 255) if (row + col) % 2 == 0 else (0, 0, 0)
                        tile_rect = pygame.Rect(
                            finish_area.left + col * tile_size,
                            finish_area.top + row * tile_size,
                            tile_size,
                            tile_size
                        )
                        pygame.draw.rect(screen, color, tile_rect)

                # Draw animated rainbow ramps
                t = frame_counter / FRAMERATE
                for shape in space.shapes:
                    if isinstance(shape, pymunk.Segment) and hasattr(shape, 'gradient_start'):
                        x1, y1 = shape.gradient_start
                        x2, y2 = shape.gradient_end
                        steps = int(math.hypot(x2 - x1, y2 - y1) // 3)  # thicker ramp look
                        for i in range(steps):
                            p = i / steps
                            x = x1 + (x2 - x1) * p
                            y = y1 + (y2 - y1) * p
                            hue = (p + t * 0.2) % 1.0
                            r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 1, 1)]
                            pygame.draw.circle(screen, (r, g, b), (int(x), int(y)), 5)

                # Draw pinwheels in white
                for shape in space.shapes:
                    if isinstance(shape, pymunk.Segment) and shape.body.body_type == pymunk.Body.DYNAMIC:
                        a = shape.a.rotated(shape.body.angle) + shape.body.position
                        b = shape.b.rotated(shape.body.angle) + shape.body.position
                        pygame.draw.line(screen, (255, 255, 255), a, b, 4)

                # Draw bumpers with animated gradient
                draw_bumpers()

                # Update and draw gradient trails
                for marble in marbles:
                    marble.update_trail()
                    marble.draw_trail(screen)  # Pass screen explicitly

                # Eliminate marbles that go out of bounds
                for i in range(len(marbles) - 1, -1, -1):
                    marble = marbles[i]
                    if marble.body.position.y > HEIGHT + 100 or marble.body.position.x < -100 or marble.body.position.x > WIDTH + 100:
                        wall_x = 50 if marble.body.position.x < 0 else WIDTH - 50 if marble.body.position.x > WIDTH else marble.body.position.x
                        spawn_elimination_burst(wall_x, marble.body.position.y, marble.color)
                        space.remove(marble.body, marble.shape)
                        del marbles[i]

                # Draw marbles
                for marble in marbles:
                    marble.draw(screen, frame_counter, FRAMERATE)  # Pass FRAMERATE explicitly

                # Check for winner
                if winner is None:
                    if check_for_winner():
                        winner_time = frame_counter / FRAMERATE  # Use frame_counter

                # Update and draw confetti
                confetti_manager.update_confetti(dt)  # Update confetti
                confetti_manager.draw_confetti(screen)  # Draw confetti

                # Draw elimination effects
                update_elimination_effects(dt)
                draw_elimination_effects()

                # End simulation 10 seconds after a winner is determined
                if winner_time is not None:
                    # Always draw winner message if available
                    font = pygame.font.SysFont("Arial", 30)
                    text = font.render("Winner: {}".format(winner["name"]), True, winner["rgb"])
                    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, 290))
                    if (frame_counter / FRAMERATE - winner_time) > 10:  # Use frame_counter
                        running = False

                pygame.display.flip()

                # Capture frame for recording
                frame = pygame.surfarray.array3d(screen)
                frame = frame.swapaxes(0, 1)  # Convert to (height, width, channels)
                recording_frames.append(frame)

            pygame.time.wait(3000)

            # Automatically save the recording for this run
            save_recording(winner_time)

        finally:
            close_midi()  # Ensure MIDI is closed properly
            audio_recording = []  # Clear audio recording at the end of the simulation

    pygame.quit()

if __name__ == '__main__':
    main()
