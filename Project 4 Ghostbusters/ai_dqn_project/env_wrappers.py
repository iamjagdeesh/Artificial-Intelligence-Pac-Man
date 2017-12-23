import numpy as np
import gym
from gym import spaces
from collections import deque
from PIL import Image
from gym.wrappers import SkipWrapper

class RenderEnv(gym.Wrapper):
    """
    Throw out the standard env observation and use a render instead
    """
    def __init__(self, env):
        super(RenderEnv, self).__init__(env)
        self.observation_space = spaces.Box(low=0., high=1., shape=(600, 400, 1))

    def get_render(self):
        render = self.env.render(mode='rgb_array')
        return render

    def _step(self, action):
        obs, rew, done, info = self.env._step(action)
        render = self.get_render()
        return render, rew, done, info

    def _reset(self):
        self.env._reset()
        return self.get_render()

#
# The following wrappers come from:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
#
class ClipRewardEnv(gym.RewardWrapper):

    def _reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.res = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.res, self.res, 1))

    def _observation(self, obs):
        frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        frame = np.array(Image.fromarray(frame).resize((self.res, self.res),
            resample=Image.BILINEAR), dtype=np.uint8)
        return frame.reshape((self.res, self.res, 1))

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        assert shp[2] == 1  # can only stack 1-channel frames
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], k))

    def _reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k): self.frames.append(ob)
        return self._observation()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)

#
# If you try the challenge problem training from images consider using the following
# environment wrapper or similar
#
def wrap_env(env):
    # Turns standard env into env trained from pixels
    env = RenderEnv(env)
    # Applys an action for k frames
    env = SkipWrapper(4)(env)
    # Reduces frame to 84x84 per DeepMind Atari
    env = WarpFrame(env)
    # Stack frames to maintain Markov property
    env = FrameStack(env, 4)
    # Maybe clip rewards but probably not nessary
    # env = ClipRewardEnv(env)
    return env
