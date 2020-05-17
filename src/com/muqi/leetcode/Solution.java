package com.muqi.leetcode;

import java.util.HashMap;
import java.util.Map;

/**
 * @author muqi
 * @since 2020/4/28 18:44
 */
public class Solution {
    // 1
    public int[] twoSum(int[] nums, int target) {
        Map<Integer,Integer> map = new HashMap<>();
        int [] result = new int[2];
        for(int i=0;i<nums.length;i++) {
            map.put(nums[i],i);
        }
        for(int i=0;i<nums.length;i++) {
            int t = target - nums[i];
            if(map.containsKey(t) && map.get(t) != i){
                result[0] = i;
                result[1] = map.get(t);
            }
        }
        return result;
    }

    // 2
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int f = 0;
        ListNode result = new ListNode(0);
        ListNode r = result;
        while(l1 != null || l2 != null || f != 0) {
            int val1 = l1 != null ? l1.val : 0;
            int val2 = l2 != null ? l2.val : 0;
            ListNode t = new ListNode(val1 + val2 + f);
            if(t.val > 9) {
                t.val %=10;
                f=1;
            }
            else f = 0;
            if(l1 != null) l1 = l1.next;
            if(l2 != null) l2 = l2.next;
            r.next = t;
            r = r.next;
        }
        return result.next;
    }
}
