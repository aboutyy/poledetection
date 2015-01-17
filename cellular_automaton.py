#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by You Li on 2014-12-11 0011
import random


class Cell:
    def __init__(self, neighbors, normal, currstate=0, prestate=0):
        self.prev_state = prestate
        self.state = currstate
        self.neighbors_indices = neighbors
        self.normal = normal

    def get_state(self):
        return self.state

    ## Subclasses will override this method in order to
    ## update the appearance of the shape that represents
    ## a cell.
    def set_state(self, state):
        self.state = state
        if state == 1:
            self.lived += 1

    def get_prev_state(self):
        return self.prev_state

    def set_prev_state(self, state):
        self.prev_state = state

    def copy_state(self):
        self.prev_state = self.state

    def get_num_lived(self):
        return self.lived

    def get_normal(self):
        return self.normal

    def set_normal(self, value):
        self.normal = value


class Automata:
    def __init__(self, numcells, init_states, neighbors_indices, normal_angle):
        self.num_cells = numcells
        self.cells = []
        # Make a (column) list.
        count = 0
        self.angle_threshold = normal_angle
        while count < self.num_cells:
            temp_cell = Cell(neighbors_indices[count], init_states[count])
            self.cells.append(temp_cell)

    ## Subclasses will override this method so that
    ## instances of their subclass of Cell will be used by
    ## the automata.
    def get_a_cell(self, chance_of_life):
        return Cell(chance_of_life)

    ## Subclasses will override this method in order to
    ## position the shapes that represent the cells.
    def init_graphics(self):
        print("Warning: Automata.initGraphics() is not implemented!")

    def get_all_cells(self):
        return self.cells

    def count_living(self):
        """
        计算生长点个数，这里对应地面点
        """
        num_living = 0.0
        for cell in self.cells:
            num_living += cell.getState()
        return num_living

    ## Subclasses can override this method in order to apply
    ## their own custom rules of life.
    def apply_rules_of_life(self, cell, live_neighbors):
        if cell.prev_state == 1:
            if live_neighbors == 2 or live_neighbors == 3:
                cell.set_state(1)
            else:
                cell.set_state(0)
        if cell.prevState == 0:
            if live_neighbors == 3:
                cell.set_state(1)
            else:
                cell.set_state(0)

    def next_generation(self):
        # Move to the "next" generation
        for cell in self.cells:
            cell.copyState()
        for cell in self.cells:
            if cell.prev_state != 1:
                if len(cell.neighbors_indices) > 0:
                    for indice in cell.neighbors_indices:
                        if cell.normal * self.cells[indice].normal < self.angle_threshold:
                            cell.set_state(1)
                            break

if __name__ == '__main__':
    pass