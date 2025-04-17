import pymunk
import pygame
import math
from common.config import MARBLE_RADIUS, TRAIL_LENGTH, COLORS

class Marble:
    def __init__(self, index, start_x, start_y, space):  # Add space parameter
        self.index = index
        self.color = COLORS[index]["rgb"]
        self.name = COLORS[index]["name"]
        self.trail = []
        self.body = pymunk.Body(mass=1, moment=10)
        self.body.position = (start_x, start_y + index * (MARBLE_RADIUS * 2))
        self.shape = pymunk.Circle(self.body, MARBLE_RADIUS)
        self.shape.elasticity = 0.6
        self.shape.color = self.color + (255,)  # Add alpha channel
        space.add(self.body, self.shape)  # Use the passed space

    def update_trail(self):
        x, y = self.body.position
        self.trail.append((x, y))
        if len(self.trail) > TRAIL_LENGTH:
            self.trail.pop(0)

    def draw_trail(self, screen):  # Accept screen as a parameter
        for j, (x, y) in enumerate(self.trail):
            alpha = int(150 * (j + 1) / TRAIL_LENGTH)
            surf = pygame.Surface((MARBLE_RADIUS * 2, MARBLE_RADIUS * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*self.color, alpha), (5, 5), MARBLE_RADIUS)
            screen.blit(surf, (x - MARBLE_RADIUS / 1.5, y - MARBLE_RADIUS / 1.5))  # Use passed screen

    def draw_halo(self, screen, frame_counter, framerate):  # Accept framerate as a parameter
        t = frame_counter / framerate  # Use the passed framerate
        speed = math.hypot(*self.body.velocity)
        pulse = 0.004 * speed + 0.15 * math.sin(t * 6) + 1
        max_radius = int(20 * pulse)
        halo_surface = pygame.Surface((max_radius * 2, max_radius * 2), pygame.SRCALPHA)
        for r in range(max_radius, 0, -1):
            alpha = int(150 - 150 * (r / max_radius) ** 2)
            pygame.draw.circle(halo_surface, (*self.color, alpha), (max_radius, max_radius), r)
        screen.blit(halo_surface, (self.body.position.x - max_radius, self.body.position.y - max_radius))  # Use passed screen

    def draw(self, screen, frame_counter, framerate):  # Accept framerate as a parameter
        self.draw_halo(screen, frame_counter, framerate)
        x, y = self.body.position
        darker_rgb = tuple(max(0, int(c * 0.5)) for c in self.color)
        pygame.draw.circle(screen, darker_rgb, (int(x), int(y)), MARBLE_RADIUS + 2)  # Use passed screen
        pygame.draw.circle(screen, self.color, (int(x), int(y)), MARBLE_RADIUS)
        shine_pos = (int(x - MARBLE_RADIUS // 3), int(y - MARBLE_RADIUS // 3))
        pygame.draw.circle(screen, (255, 255, 255), shine_pos, 2)
