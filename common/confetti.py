import pygame
import random
import math

class ConfettiManager:
    def __init__(self, width, height, colors, confetti_count):
        self.width = width
        self.height = height
        self.colors = colors
        self.confetti_count = confetti_count
        self.particles = []

    def spawn_confetti(self):
        for _ in range(self.confetti_count):
            x, y = self.width // 2, self.height // 2.5
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0, 300)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - 200
            color = random.choice(self.colors)
            radius = random.randint(2, 4)
            particle = {
                'pos': [x, y],
                'vel': [vx, vy],
                'radius': radius,
                'color': color["rgb"]
            }
            self.particles.append(particle)

    def update_confetti(self, dt):
        for particle in self.particles:
            particle['vel'][1] += 300 * dt  # gentle gravity
            particle['pos'][0] += particle['vel'][0] * dt
            particle['pos'][1] += particle['vel'][1] * dt

    def draw_confetti(self, screen):
        for particle in self.particles:
            pygame.draw.circle(screen, particle['color'], (int(particle['pos'][0]), int(particle['pos'][1])), particle['radius'])
