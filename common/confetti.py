import pygame
import random
import math
from common.config import WIDTH, HEIGHT, CONFETTI_COUNT, COLORS

confetti_particles = []

def spawn_confetti():
    for _ in range(CONFETTI_COUNT):
        x, y = WIDTH // 2, HEIGHT // 2.5
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0, 300)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed - 200
        color = random.choice(COLORS)
        radius = random.randint(2, 4)
        particle = {
            'pos': [x, y],
            'vel': [vx, vy],
            'radius': radius,
            'color': color["rgb"]
        }
        confetti_particles.append(particle)

def update_confetti(dt):
    for particle in confetti_particles:
        particle['vel'][1] += 300 * dt  # gentle gravity
        particle['pos'][0] += particle['vel'][0] * dt
        particle['pos'][1] += particle['vel'][1] * dt

def draw_confetti(screen):
    for particle in confetti_particles:
        pygame.draw.circle(screen, particle['color'], (int(particle['pos'][0]), int(particle['pos'][1])), particle['radius'])
