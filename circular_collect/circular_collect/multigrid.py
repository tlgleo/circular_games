import math
import gym
from enum import IntEnum
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from .rendering import *
from .window import Window
import numpy as np

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    'red': np.array([220, 100, 70]),
    'green': np.array([125, 170, 85]),
    'blue': np.array([50, 110, 185]),
    'yellow': np.array([245, 195, 65]),
    'purple': np.array([150, 110, 200]),
    'grey': np.array([100, 100, 100])
}

COLOR_NAMES = sorted(list(COLORS.keys()))

class World_4:

    encode_dim = 8
    n_agents = 4
    normalize_obs = 0.2
    # to normalise observation (different coeff by column because not the same range of values)
    # e.g. color from 0 to 5 -> 0.2
    normalize_obs_per_dim = [0.33, 0.2, 0.33, 1.0]

    # Used to map colors to integers
    COLOR_TO_IDX = {
        'red': 0,
        'green': 1,
        'blue': 2,
        'yellow': 3,
        'purple': 4,
        'grey': 5
    }

    IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

    # Map of object type to integers
    OBJECT_TO_IDX = {
        'empty': 0,
        'wall': 1,
        'ball': 2,
        'agent': 3,
    }
    IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))


# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, world, type, color):
        assert type in world.OBJECT_TO_IDX, type
        assert color in world.COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def encode(self, world, current_agent=False):
        """Encode the a description of this object as a 3-tuple of integers"""
        pass

    @staticmethod
    def decode(type_idx, color_idx, state):
        assert False, "not implemented"

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError



class Wall(WorldObj):
    def __init__(self, world, color='grey'):
        super().__init__(world, 'wall', color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

    def encode(self, world, current_agent=False):
        """Encode the a description of this object as a tuple of integers"""
        return tuple([0.0] * world.encode_dim)


class Ball(WorldObj):
    def __init__(self, world, index=0, reward=1):
        super(Ball, self).__init__(world, 'ball', world.IDX_TO_COLOR[index])
        self.index = index
        self.reward = reward

    def can_pickup(self):
        return True

    def encode(self, world, current_agent=False, n_agents = 4):
        """Encode the a description of this object as a tuple of integers"""
        vect = [0.0] * world.encode_dim
        vect[self.index] = 1
        return tuple(vect)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.2), COLORS[self.color])




class Agent(WorldObj):
    def __init__(self, world, index=0, view_size=7):
        super(Agent, self).__init__(world, 'agent', world.IDX_TO_COLOR[index])
        self.pos = None
        self.dir = None
        self.index = index
        self.view_size = view_size
        self.terminated = False
        self.started = True
        self.paused = False


    def render(self, img):
        c = COLORS[self.color]
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )
        # Rotate the agent based on its direction
        #tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta= -0.5 * math.pi)
        #fill_coords(img, tri_fn, c)
        fill_coords(img, point_in_semicircle(0.5, 0.9, 0.37), c)
        fill_coords(img, point_in_circle(0.5, 0.35, 0.15), c)

    def can_overlap(self):
        return False

    def encode(self, world, current_agent=False, n_agents=4):
        """Encode the a description of this object as a tuple of integers"""
        vect = [0.0] * world.encode_dim
        vect[self.index + world.n_agents] = 1
        return tuple(vect)

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.dir >= 0 and self.dir < 4
        return DIR_TO_VEC[self.dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.pos + self.dir_vec

    def next_pos(self, action):
        """
        Get the position of the cell after taking one action (right, down, left, up)
        """
        return self.pos + DIR_TO_VEC[action]


    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.view_size
        hs = self.view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx * lx + ry * ly)
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        # Facing right
        if self.dir == 0:
            topX = self.pos[0]
            topY = self.pos[1] - self.view_size // 2
        # Facing down
        elif self.dir == 1:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1]
        # Facing left
        elif self.dir == 2:
            topX = self.pos[0] - self.view_size + 1
            topY = self.pos[1] - self.view_size // 2
        # Facing up
        elif self.dir == 3:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1] - self.view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.view_size
        botY = topY + self.view_size

        return (topX, topY, botX, botY)

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.view_size or vy >= self.view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        self.grid = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def horz_wall(self, world, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type(world))

    def vert_wall(self, world, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type(world))

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, world, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                        y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall(world)

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
            cls,
            world,
            obj,
            highlights=[],
            tile_size=TILE_PIXELS,
            subdivs=3
    ):
        """
        Render a tile and cache the result
        """

        key = (*highlights, tile_size)
        key = obj.encode(world) + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj != None:
            obj.render(img)

        # Highlight the cell  if needed
        if len(highlights) > 0:
            for h in highlights:
                highlight_img(img, color=COLORS[world.IDX_TO_COLOR[h%len(world.IDX_TO_COLOR)]])

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
            self,
            world,
            tile_size,
            highlight_masks=None
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                # agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = Grid.render_tile(
                    world,
                    cell,
                    highlights=[] if highlight_masks is None else highlight_masks[i, j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, world, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, world.encode_dim), dtype='float32')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = world.OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0
                        if world.encode_dim > 3:
                            array[i, j, 3] = 0

                    else:
                        array[i, j, :] = v.encode(world)

        return array

    def encode_for_agents(self, world, agent_pos, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """
        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, world.encode_dim), dtype='float32')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, :] = 0

                    else:
                        array[i, j, :] = v.encode(world, current_agent=np.array_equal(agent_pos, (i, j)))

        return array


class Actions:
    available=['still', 'left', 'right', 'forward']
    still = 0
    # Turn left, turn right, move forward
    left = 1
    right = 2
    forward = 3


class MultiGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    # Enumeration of possible actions

    def __init__(
            self,
            grid_size=None,
            width=None,
            height=None,
            max_steps=100,
            seed=2,
            agents=None,
            partial_obs=False,
            agent_view_size=7,
            actions_set=Actions,
            objects_set=World_4
    ):
        self.agents = agents

        # Does the agents have partial or full observation?
        self.partial_obs = partial_obs

        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        self.actions = actions_set

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions.available))

        self.objects=objects_set

        if partial_obs:
            self.observation_space = spaces.Box(
                low=-1,
                high=1,
                shape=(agent_view_size, agent_view_size, self.objects.encode_dim)
            )

        else:
            self.observation_space = spaces.Box(
                low=-1,
                high=1,
                shape=(width, height, self.objects.encode_dim)

            )

        self.ob_dim = np.prod(self.observation_space.shape)
        self.ac_dim = self.action_space.n

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps

        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

    def reset(self):
        pass

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count


    def _gen_grid(self, width, height):
        assert False, "_gen_grid needs to be implemented by each environment"

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        pass


    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.randint(low, high)


    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]


    def place_obj(self,
                  obj,
                  top=None,
                  size=None,
                  reject_fn=None,
                  max_tries=math.inf
                  ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(
            self,
            agent,
            top=None,
            size=None,
            rand_dir=True,
            max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        agent.pos = None
        pos = self.place_obj(agent, top, size, max_tries=max_tries)
        agent.pos = pos
        agent.init_pos = pos

        if rand_dir:
            agent.dir = self._rand_int(0, 4)

        agent.init_dir = agent.dir

        return pos

    def step(self, actions):
        pass


    def render(self, mode='human', close=False, highlight=False, tile_size=TILE_PIXELS, save_fig = False, fig_name = 'render.png'):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            self.window = Window('Circular_collect')
            self.window.show(block=False)


        # Render the whole grid
        img = self.grid.render(
            self.objects,
            tile_size,
            highlight_masks= None
        )

        if not save_fig:
            #print('not save')
            fig_name = None

        if mode == 'human':
            self.window.show_img(img, fig_name=fig_name)

        return img
