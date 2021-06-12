import pygame
import random
import neat
from math import hypot

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 30

PLAYER_IMG = pygame.image.load("assets/drone.png")
PLAYER_IMG_HEIGHT = PLAYER_IMG.get_height()
PLAYER_IMG_WIDTH = PLAYER_IMG.get_width()

GRAVITY = 0.5
DRAG = 0.5

# Initialize pygame
pygame.init()
pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 30)


# Player
class Drone:

    def __init__(self):
        self.img = PLAYER_IMG
        # coordinates
        self.x = 340
        self.y = 200
        # velocity
        self.y_vel = 0
        self.x_vel = 0
        # velocity boundaries
        self.y_vel_up_boundary = -15
        self.y_vel_down_boundary = 15
        # acceleration
        self.y_acc = 0
        self.x_acc = 0
        # rotation
        self.rotation_angle = 0

        self.cargo_rect = pygame.Rect(self.x + (PLAYER_IMG_WIDTH / 2) - 5, self.y + (PLAYER_IMG_HEIGHT / 2) + 6, 10, 10)

        self.is_collided = False
        self.is_dead = False

    def go_up(self):
        self.y_acc = -1

    def go_right(self):
        self.x_acc = 1
        self.rotation_angle += -5

    def go_left(self):
        self.x_acc = -1
        self.rotation_angle += 5

    def move(self):
        # y movement
        self.y_vel += self.y_acc + GRAVITY
        if self.y_vel < self.y_vel_up_boundary:
            self.y_vel = self.y_vel_up_boundary
        elif self.y_vel > self.y_vel_down_boundary:
            self.y_vel = self.y_vel_down_boundary
        self.y += self.y_vel
        self.y_acc = 0
        # x movement
        self.x_vel += self.x_acc
        if self.x_vel > 0:
            self.x_vel -= DRAG
        elif self.x_vel < 0:
            self.x_vel += DRAG
        self.x += self.x_vel
        self.x_acc = 0
        # rotation
        if self.rotation_angle > 30:
            self.rotation_angle = 30
        elif self.rotation_angle < -30:
            self.rotation_angle = -30
        if self.rotation_angle > 0:
            self.rotation_angle -= 2
        elif self.rotation_angle < 0:
            self.rotation_angle += 2

    def draw(self, screen):
        self.cargo_rect = pygame.Rect(self.x + (PLAYER_IMG_WIDTH / 2) - 5, self.y + (PLAYER_IMG_HEIGHT / 2) + 6, 10, 10)
        rotated_img = pygame.transform.rotate(self.img, self.rotation_angle)
        new_rect = rotated_img.get_rect(
            center=self.img.get_rect(topleft=(self.x, self.y)).center)
        screen.blit(rotated_img, new_rect.topleft)

    def draw_rectangles(self, screen):
        cargo_rect = pygame.Rect(self.x + (PLAYER_IMG_WIDTH / 2) - 5, self.y + (PLAYER_IMG_HEIGHT / 2) + 6, 10, 10)
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(self.x, self.y, PLAYER_IMG_WIDTH, PLAYER_IMG_HEIGHT))
        pygame.draw.rect(screen, (0, 255, 0), cargo_rect)

    def draw_line_to_target(self, screen, target):
        pygame.draw.line(screen, (0, 0, 255), self.cargo_rect.center, target.center())
        distance = self.distance_to_target(target)
        if distance < 5 and not self.is_collided:
            self.is_collided = True

    def distance_to_target(self, target):
        return hypot(target.center()[0] - self.cargo_rect.center[0], target.center()[1] - self.cargo_rect.center[1])


class Target:

    def __init__(self):
        self.x = random.randint(10, SCREEN_WIDTH - 10)
        self.y = random.randint(10, SCREEN_HEIGHT - 10)
        self.width = 5
        self.color = (0, 255, 0)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.width)

    def center(self):
        return self.x + (self.width / 2), self.y + (self.width / 2)


def draw_text(screen, successfull):
    textsurface = myfont.render(f'Success: {successfull}', False, (255, 0, 0))
    screen.blit(textsurface, (0, 0))


def draw_screen(screen, drone, target, successfull):
    screen.fill((0, 0, 0))

    drone.draw_rectangles(screen=screen)
    drone.draw_line_to_target(screen=screen, target=target)
    drone.draw(screen=screen)

    target.draw(screen=screen)

    draw_text(screen=screen, successfull=successfull)


def main():
    clock = pygame.time.Clock()

    target = Target()
    drone = Drone()
    successfull_drones = 0

    # create the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    # Title and icon
    pygame.display.set_caption("Drone - NEAT")

    # Game Loop
    running = True
    while running:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()

            K_UP_PRESSED = False
            K_RIGHT_PRESSED = False
            K_LEFT_PRESSED = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    K_UP_PRESSED = True
                if event.key == pygame.K_LEFT:
                    K_LEFT_PRESSED = True
                if event.key == pygame.K_RIGHT:
                    K_RIGHT_PRESSED = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    K_UP_PRESSED = False
                if event.key == pygame.K_LEFT:
                    K_LEFT_PRESSED = False
                if event.key == pygame.K_RIGHT:
                    K_RIGHT_PRESSED = False

        if K_UP_PRESSED:
            drone.go_up()
        if K_RIGHT_PRESSED:
            drone.go_right()
        if K_LEFT_PRESSED:
            drone.go_left()

        drone.move()
        print(drone.cargo_rect.center)
        if drone.x > 810 or drone.x < -10 or drone.y > 610 or drone.y < -10:
            drone.is_dead = True

        draw_screen(screen=screen, drone=drone, target=target, successfull=successfull_drones)

        pygame.display.update()


if __name__ == '__main__':
    main()
