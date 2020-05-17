package com.muqi.leetcode.test;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

/**
 * @author muqi
 * @since 2020/5/15 20:41
 *
 * 30
 */
public class MinStack {

    LinkedList<Integer> v = new LinkedList<>();
    int index;
    int min;
    public MinStack() {
        this.index = -1;
        this.min = Integer.MAX_VALUE;
    }

    public void push(int x) {
        v.add(x);
        index++;
        if(x < min) min = x;
    }

    public void pop() {
        int x = v.remove(index--);
        if(min == x) {
            min = Integer.MAX_VALUE;
            if(!v.isEmpty()) {
                v.forEach(i->{
                    if(i<min) min=i;
                });
            }
        }
    }

    public int top() {
        return v.get(index);
    }

    public int min() {
        return min;
    }
}
