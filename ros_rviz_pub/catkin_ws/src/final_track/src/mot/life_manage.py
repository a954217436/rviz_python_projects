""" a finite state machine to manage the life cycle
    states: 
      - new:    new object
      - stable: stable object
      - lose:   without high score association, about to die
      - delete: may it eternal peace
"""
import numpy as np


class TrackState:
    new     = 0
    stable  = 1
    lose    = 2
    delete  = 3


default_trans_cfg = {"NEW2STABLE"    : 1,
                     "STABLE2LOSE"   : 2,
                     "LOSE2DELETE"   : 3,
                     "NEW2DELETE"    : 2,
                     "LOSE2STABLE"   : 1}


class LifeManager(object):
    def __init__(self, trans_cfg = default_trans_cfg):
        self.trans_cfg = trans_cfg

        self.hits = 0           # number of total hits including the first detection
        self.hit_streak = 0     # number of continuing hit considering the first detection
        self.age = 0
        # self.asso_score = 0
        self.time_since_update = 0

        self.state = TrackState.new
        self.first_continuing_hit = 1
        self.still_first = True

    # def state(self):
    #     return self.state

    def predict(self):
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.state_transition()


    def update(self):
        # self.asso_score = asso_score
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.state_transition()

    
    def state_transition(self):
        if self.state == TrackState.new:
            if self.hit_streak >= self.trans_cfg["NEW2STABLE"]:
                self.state = TrackState.stable
            if self.time_since_update >= self.trans_cfg["NEW2DELETE"]:
                self.state = TrackState.delete

        if self.state == TrackState.stable:
            if self.time_since_update >= self.trans_cfg["STABLE2LOSE"]:
                self.state = TrackState.lose

        if self.state == TrackState.lose:
            if self.time_since_update >= self.trans_cfg["LOSE2DELETE"]:
                self.state = TrackState.delete
            if self.hit_streak >= self.trans_cfg["LOSE2STABLE"]:
                self.state = TrackState.stable
        
    def new(self):
        return self.state == TrackState.new

    def stable(self):
        return self.state == TrackState.stable

    def lose(self):
        return self.state == TrackState.lose

    def delete(self):
        return self.state == TrackState.delete

    def state_string(self):
        info_str = "ht=%d,ag=%d,hs=%d,tsu=%d"%(self.hits, self.age, self.hit_streak, self.time_since_update)

        if self.state == TrackState.new:
            return 'new_{}'.format(info_str)
        if self.state == TrackState.stable:
            return 'sta_{}'.format(info_str)
        if self.state == TrackState.lose:
            return 'los_{}'.format(info_str)
        if self.state == TrackState.delete:
            return 'del_{}'.format(info_str)


