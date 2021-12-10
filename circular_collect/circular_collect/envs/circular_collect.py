from circular_collect.multigrid import *
from circular_collect.utils import *
from copy import deepcopy

class Actions_1:
    available=['still', 'left', 'right', 'forward']

    # Turn left, turn right, move forward
    still = 0
    left = 1
    right = 2
    forward = 3

class Actions_2:
    available=['right', 'down', 'left', 'up', 'still']

    # go right, down, left and up OR stay still

    right = 0
    down = 1
    left = 2
    up = 3
    still = 4



R_STEP = 0

R_OWN = 2 - R_STEP
R_OTHER = -1 - R_STEP

#radius for distance distribution
RAD_DIST = 0.25

#probas for balls position around players
#with a 4 Players cycle and four 3 players
CIRC_PROBAS_4P_SC = \
    [[0.25, 0.50, 0.25, 0.00],
     [0.00, 0.25, 0.50, 0.25],
     [0.25, 0.00, 0.25, 0.50],
     [0.50, 0.25, 0.00, 0.25]
    ]

CIRC_PROBAS_4P_P = \
    [[0.5, 0.2, 0.2, 0.1],
     [0.1, 0.5, 0.2, 0.2],
     [0.2, 0.1, 0.5, 0.2],
     [0.2, 0.2, 0.1, 0.5]
    ]

#without cycles
CIRC_PROBAS_4P_0C = \
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0],
    ]

CIRC_PROBAS_4P_1C = \
    [[0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0, 0.0],
    ]


CIRC_PROBAS_4P_B = \
    [[0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
    ]




CIRC_PROBAS_4P_U = 0.25*np.ones([4,4])
CIRC_PROBAS_5P_0C = np.eye(5)


IDENT_TO_COORD = [(0,0),(1,0),(1,1),(0,1)]
COORD_TO_IDENT = [[0,3],[1,2]]


class CircularCollect(MultiGridEnv):
    """
    Environment in which the agents have to collect the balls
    """

    def __init__(
        self,
        size=None,
        width=None,
        height=None,
        num_balls=[],
        agents_index = [0,1,2,3],
        agents_pos = None,
        balls_index=[],
        balls_reward=[],
        zero_sum = False,
        view_size=7,
        actions_set = Actions_2,
        prob_matrix = None,
        n_agents = 4,
        type_actions = 2,
        time_rem_ball = 5
    ):
        self.num_balls = num_balls  # list of number of balls per agent in the grid
        self.balls_index = balls_index #
        self.balls_reward = balls_reward
        self.zero_sum = zero_sum
        self.agents_pos = agents_pos

        self.world = World_4
        self.balls_on_grid = [0]*len(agents_index)
        self.balls_in_room = [0]*len(agents_index)
        self.prob_matrix = prob_matrix
        self.n_agents = n_agents
        self.type_actions = type_actions

        self.balls_per_color = False
        self.balls_per_agent = self.num_balls



        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        self.time_rem_ball = time_rem_ball
        self.balls = dict()
        self.coord_rooms = []


        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps= 10000,
            agents=agents,
            agent_view_size=view_size,
            actions_set = actions_set
        )



    def reset(self, seed = None, num_balls = None):

        #print('reset env')
        if num_balls is not None:
            self.num_balls = num_balls
            self.balls_per_agent = num_balls

        if seed is not None:
            self.seed = seed

        self.get_coord_rooms()
        self.balls = dict()
        self.balls_on_grid = [0] * len(self.agents)
        self.balls_in_room = [0] * len(self.agents)
        self._gen_grid(self.width, self.height)


        # These fields should be defined by _gen_grid
        for a in self.agents:
            assert a.pos is not None
            assert a.dir is not None

        # Item picked up, being carried, initially nothing
        for a in self.agents:
            a.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        return self._get_obs()

    def replicate(self, env, seed):
        # create a new environment from another env and with new seed (for rollouts)
        self.balls = deepcopy(env.balls)
        self.balls_on_grid = deepcopy(env.balls_on_grid)
        self.balls_in_room = deepcopy(env.balls_in_room)
        self.num_balls = deepcopy(env.num_balls)
        self.balls_per_agent = deepcopy(env.balls_per_agent)
        self.grid = deepcopy(env.grid)
        self.seed(seed)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        # Randomize the player start position and orientation
        for i,a in enumerate(self.agents):
            if self.agents_pos == None:
                self.place_agent(a)
            else:
                self.place_agent(a, self.agents_pos[i], size = (1,1))

        self._update_balls()

    def get_coord_rooms(self):
        for x in range(4):
            (i,j) = IDENT_TO_COORD[x]
            x = (1+int(self.width/2)*i, int(self.width/2) + i*int(self.width/2))
            y = (1+int(self.height/2)*j, int(self.height/2) + j*int(self.height/2))
            self.coord_rooms.append([x,y])

    def sample_position_room(self,i):
        return (self._rand_int(*(self.coord_rooms)[i][0]), self._rand_int(*(self.coord_rooms)[i][1]))

    def is_in_room(self, x, y, ident):
        return True

    def pos2room_index(self, i, j):
        return COORD_TO_IDENT[int(2*i/self.width)][int(2*j/self.height)]

    def room_index2zone(self, i):
        return []


    def _update_balls(self):
        self._remove_balls()
        for i,a in enumerate(self.agents):
            m = self.balls_per_agent[i] - self.balls_in_room[i]

            for _ in range(m):
                probs = self.prob_matrix[i] #probs to choose a color for ball close to player i
                i_C = np.random.choice(len(self.agents), p=probs) #choice of a color

                max_tries = 100
                for _ in range(max_tries):
                    #sample a position near pos_a with radial distribution
                    pos = self.sample_position_room(i)
                    if self.grid.get(*pos) == None:
                        self._add_ball(pos, i_C)
                        self.balls_in_room[i] += 1
                        break


    def _time_to_remove_ball(self, t):
        return expo_dist(t, self.time_rem_ball)

    def _remove_balls(self):

        remove_list = []
        for (i,j) in self.balls:
            self.balls[(i,j)][1] += 1
            if self._time_to_remove_ball(self.balls[(i,j)][1]):
                # remove ball
                remove_list.append((i, j))
                self.balls_in_room[self.pos2room_index(i,j)] -= 1
                self.grid.set(i,j, None)
        for c in remove_list:
            del self.balls[c]


    def _add_ball(self, pos, index, size = (1,1)):
        self.place_obj(Ball(self.world, index, reward=0), pos, size=size)
        self.balls[pos] = [index, 1]


    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if fwd_cell.index in [0, self.agents[i].index]:
                    fwd_cell.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
                    self._reward(i, rewards, fwd_cell.reward)

    def _get_obs(self):

        obs = [self.grid.encode_for_agents(self.objects, self.agents[i].pos) for i in range(len(self.agents))]

        #normalize_obs_per_dim_np = np.full([self.width, self.height, self.objects.encode_dim], np.array(self.objects.normalize_obs_per_dim))
        #obs = [np.multiply(normalize_obs_per_dim_np, ob) for ob in obs]

        return obs


    def step(self, actions):
        self.step_count += 1
        picked_ball = False
        order = np.random.permutation(len(actions))

        rewards = np.zeros(len(actions)) + R_STEP
        done = False

        for i in order:

            if self.agents[i].terminated or self.agents[i].paused or not self.agents[i].started or actions[
                i] == self.actions.still :
                continue

            # Get the position in front of the agent
            fwd_pos = self.agents[i].next_pos(actions[i])

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            if fwd_cell is not None:
                if fwd_cell.type == 'ball':
                    if fwd_cell.index == self.agents[i].index:
                        # player i receives R_OWN for its own coin
                        rewards[i] += R_OWN
                        self.balls_in_room[i] -= 1
                        del self.balls[(fwd_pos[0],fwd_pos[1])]
                        picked_ball = True
                    else:
                        # player i receives R_OTHER for its another coin
                        rewards[i] += R_OTHER
                        # player of the coin receives R_OWN for its another coin
                        rewards[fwd_cell.index] += R_OWN
                        self.balls_in_room[i] -= 1
                        del self.balls[(fwd_pos[0],fwd_pos[1])]
                        picked_ball = True

            if fwd_cell is None or fwd_cell.can_overlap():
                self.grid.set(*fwd_pos, self.agents[i])
                self.grid.set(*self.agents[i].pos, None)
                self.agents[i].pos = fwd_pos


        if self.step_count >= self.max_steps:
            done = True


        self._update_balls()

        obs = self._get_obs()

        return obs, rewards, done, {}


class CircularCollect4players(CircularCollect):
    """
        Environment in which the agents have to collect the balls
        """
    def __init__(
            self,
            size = None,
            width=None,
            height=None,
            num_balls=[2,2,2,2],
            agents_pos=[(2,2), (2,6),(6,6),(6,2)],
            prob_matrix=CIRC_PROBAS_4P_1C,
            time_rem_ball = 5
    ):
        super().__init__(width=width,
                         height=height,
                         num_balls=num_balls,
                         agents_pos=agents_pos,
                         agents_index = [0,1,2,3],
                         prob_matrix=prob_matrix,
                         n_agents=4,
                         type_actions=2,
                         time_rem_ball=time_rem_ball
                         )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        self.grid.vert_wall(self.world, int(width/2), 0)
        self.grid.horz_wall(self.world, 0, int(height/2))


        # Randomize the player start position and orientation
        for i,a in enumerate(self.agents):
            if self.agents_pos == None:
                self.place_agent(a)
            else:
                self.place_agent(a, self.agents_pos[i], size = (1,1))

        self._update_balls()




class CircularCollect4players_4x4_4P_Cir(CircularCollect4players):
    def __init__(self):
        super().__init__(
            size = None,
            width=11,
            height=11,
            num_balls=[1,1,1,1],
            agents_pos=[(2,2), (7,2),(7,7),(2,7)],
            prob_matrix=CIRC_PROBAS_4P_1C
    )


class CircularCollect4players_4x4_4P_Bil(CircularCollect4players):
    def __init__(self):
        super().__init__(
            size = None,
            width=11,
            height=11,
            num_balls=[1,1,1,1],
            agents_pos=[(2,2), (7,2),(7,7),(2,7)],
            prob_matrix=CIRC_PROBAS_4P_B
    )


class CircularCollect4players_4x4_4P_Uni(CircularCollect4players):
    def __init__(self):
        super().__init__(
            size = None,
            width=11,
            height=11,
            num_balls=[1,1,1,1],
            agents_pos=[(2,2), (7,2),(7,7),(2,7)],
            prob_matrix=CIRC_PROBAS_4P_U,
            time_rem_ball=50

    )



class CircularCollect4players_4x4_4P_Uni_2(CircularCollect4players):
    def __init__(self):
        super().__init__(
            size = None,
            width=11,
            height=11,
            num_balls=[1,1,1,1],
            agents_pos=[(2,2), (7,2),(7,7),(2,7)],
            prob_matrix=CIRC_PROBAS_4P_U,
            time_rem_ball=5

    )

