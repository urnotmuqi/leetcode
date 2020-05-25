package com.muqi.leetcode.test;

import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.Queue;

/**
 * @author muqi
 * @since 2020/5/25 19:48
 *
 * 41
 */
public class MedianFinder {
    Queue<Integer> a,b;
    /** initialize your data structure here. */
    public MedianFinder() {
        a = new PriorityQueue<>();        //小根堆
        b = new PriorityQueue<>((x,y) -> (y-x));       //大根堆
    }

    public void addNum(int num) {
        if(a.size() == b.size()) {
            b.add(num);
            a.add(b.poll());
        }
        else {
            a.add(num);
            b.add(a.poll());
        }
    }

    public double findMedian() {
        double res = 0.0;
        if(a.size() == b.size()) res = (a.peek() + b.peek())*1.0/2;
        else res = a.peek()*1.0;
        return res;
    }
}
