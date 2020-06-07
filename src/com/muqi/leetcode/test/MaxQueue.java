package com.muqi.leetcode.test;

import java.util.Deque;
import java.util.LinkedList;
import java.util.Queue;

/**
 * @author muqi
 * @since 2020/6/6 17:53
 *  59-2
 */
class MaxQueue {
    Queue<Integer> queue = new LinkedList<>();
    Deque<Integer> deque = new LinkedList<>();

    public MaxQueue() {

    }

    public int max_value() {
        if(deque.isEmpty()) return -1;
        return deque.peekFirst();
    }

    public void push_back(int value) {
        queue.add(value);
        while(!deque.isEmpty() && deque.peekLast() < value) {
            deque.removeLast();
        }
        deque.addLast(value);
    }

    public int pop_front() {
        if(queue.isEmpty()) return -1;
        int num = queue.poll();
        if(num == deque.getFirst()) deque.removeFirst();
        return num;
    }
}
