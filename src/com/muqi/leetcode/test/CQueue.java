package com.muqi.leetcode.test;


import java.util.Stack;

/**
 * @author muqi
 * @since 2020/5/8 17:32
 *
 * 面试题09
 *
 */
public class CQueue {
    Stack<Integer> s1;
    Stack<Integer> s2;

    public CQueue() {
        s1 = new Stack<>();
        s2 = new Stack<>();
    }

    public void appendTail(int value) {
        s1.push(value);
    }

    public int deleteHead() {
        if(s1.empty() && s2.empty()) return -1;
        else if(s2.empty()) {
            while(!s1.empty()) {
                s2.push(s1.pop());
            }
        }
        return s2.pop();
    }
}
