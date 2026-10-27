"""Keyboard driving with the same five actions as the random agent."""

import argparse

import numpy as np
import pygame

from neural_racing_lab.envs import make_env, reset_seeded
from neural_racing_lab.random_agent import positive_int


def keyboard_action(*, left: bool, right: bool, gas: bool, brake: bool) -> int:
    """Discrete actions cannot combine gas and steering; brake has priority."""
    if brake:
        return 4
    if left != right:
        return 2 if left else 1
    return 3 if gas else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=positive_int, default=1000)
    parser.add_argument("--frames", type=positive_int, help="Exit after N frames for a smoke test")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    # rgb_array lets this module own the window and its event queue.
    env = make_env(render_mode="rgb_array", max_episode_steps=args.max_episode_steps)
    pygame.init()
    try:
        reset_seeded(env, args.seed)
        frame = env.render()
        screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
        clock = pygame.time.Clock()
        episode_return = 0.0
        episode_steps = 0
        frame_count = 0
        finished = False
        print("Arrows/WASD: drive; Space: brake; R: restart same track; Esc: quit")
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        reset_seeded(env, args.seed)
                        episode_return, episode_steps, finished = 0.0, 0, False
            if not running:
                break
            keys = pygame.key.get_pressed()
            action = keyboard_action(
                left=keys[pygame.K_LEFT] or keys[pygame.K_a],
                right=keys[pygame.K_RIGHT] or keys[pygame.K_d],
                gas=keys[pygame.K_UP] or keys[pygame.K_w],
                brake=keys[pygame.K_DOWN] or keys[pygame.K_s] or keys[pygame.K_SPACE],
            )
            if not finished:
                _, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                episode_steps += 1
                finished = terminated or truncated
                if finished:
                    print(f"Return={episode_return:.2f}, {terminated=}, {truncated=}; press R")
            frame = env.render()
            screen.blit(pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1)), (0, 0))
            pygame.display.set_caption(
                f"Neural Racing Lab | return {episode_return:.1f} | step {episode_steps}"
                + (" | FINISHED: R to restart" if finished else "")
            )
            pygame.display.flip()
            clock.tick(env.metadata["render_fps"])
            frame_count += 1
            if args.frames is not None and frame_count >= args.frames:
                break
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
