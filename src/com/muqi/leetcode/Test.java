package com.muqi.leetcode;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author muqi
 * @since 2020/4/24 19:06
 */
public class Test {
    public static void main(String[] args) {
        Solution s = new Solution();
        int[] num1 = {2,4,5,2};
        int[] num2 = {5,6,4};
        ListNode l1 = new ListNode(0);
        ListNode l2 = new ListNode(0);
        ListNode p = l1;
        ListNode q = l2;
        for(int i : num1) {
            ListNode t = new ListNode(i);
            p.next = t;
            p = p.next;
        }
        for(int i : num2) {
            ListNode t = new ListNode(i);
            q.next = t;
            q = q.next;
        }
        ListNode r = s.addTwoNumbers(l1.next,l2.next);
        while(r != null) {
            System.out.print(r.val + ",");
            r = r.next;
        }
    }

}
