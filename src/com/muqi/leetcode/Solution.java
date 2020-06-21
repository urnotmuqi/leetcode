package com.muqi.leetcode;

import java.util.*;

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

    //3
    public int lengthOfLongestSubstring(String s) {
        //set法
//        HashSet<Character> set = new HashSet<>();
//        int i=0,j=0,len=0;
//        while(i<s.length()&&j<s.length()) {
//            if(set.contains(s.charAt(j))) {
//                if(j-i>len) {
//                    len = j-i;
//                }
//                while(s.charAt(i)!=s.charAt(j)) {
//                    set.remove(s.charAt(i++));
//                }
//                set.remove(s.charAt(i++));
//            }
//            set.add(s.charAt(j++));
//        }
//        if(j-i>len) len=j-i;
//        return len;

        //map法
//        Map<Character, Integer> map = new HashMap<>();
//        int i=0,j=0,len=0;
//        while(i<s.length()&&j<s.length()) {
//            if(map.containsKey(s.charAt(j)) && map.get(s.charAt(j)) >= i) {
//                if(j-i>len) {
//                    len = j-i;
//                }
//                i = map.get(s.charAt(j)) + 1;
//            }
//            map.put(s.charAt(j),j++);
//        }
//        if(j-i>len) len=j-i;
//        return len;

        //数组法
        int[] last = new int[128];
        for(int i=0;i<128;i++) {
            last[i] = -1;
        }
        int n = s.length();
        int len = 0,start=0;
        for(int i=0;i<s.length();i++) {
            int index = s.charAt(i);
            //if(last[index] >= start) start = last[index] + 1;
            start = Math.max(start, last[index]+1);
            len = Math.max(len, i-start+1);
            last[index] = i;
        }
        return len;
    }

    //6
    public String convert(String s, int numRows) {
        //按行访问
//        int k = numRows*2-2;
//        int n = s.length();
//        int m = n%k==0?n/k:n/k+1;
//        StringBuilder res = new StringBuilder();
//        for(int i=0;i<numRows;i++) {
//            for(int j=0;j<m;j++) {
//                if(i==0 || i==numRows-1) {
//                    if(j*k+i<n) res.append(s.charAt(j*k+i));
//                }
//                else {
//                    if(j*k+i<n) res.append(s.charAt(j*k+i));
//                    if(j*k+k-i<n) res.append(s.charAt(j*k+k-i));
//                }
//            }
//        }
//        return res.toString();

        //按行排序
        if(numRows == 1) return s;
        List<StringBuilder> list = new ArrayList<>();
        for(int i=0;i<Math.min(numRows,s.length());i++) {
            list.add(new StringBuilder());
        }
        boolean goDown = false;
        int curRow = 0;
        for(char c : s.toCharArray()) {
            list.get(curRow).append(c);
            if(curRow==0 || curRow==numRows-1) goDown = !goDown;
            curRow += goDown==true ? 1 : -1;
        }
        StringBuilder res = new StringBuilder();
        for(StringBuilder sb : list) res.append(sb);
        return res.toString();
    }

    //7
    public int reverse(int x) {
        int r = 0;
        while(x != 0) {
            int t = x % 10;
            x /= 10;
            if(r > Integer.MAX_VALUE/10 || r == Integer.MAX_VALUE && t > 7) return 0;
            if(r < Integer.MIN_VALUE/10 || r == Integer.MIN_VALUE && t < -8) return 0;
            r = r*10 + t;
        }
        return r;
    }

    //9
    public boolean isPalindrome(int x) {
//        String s = String.valueOf(x);
//        for(int i=0;i<s.length()/2;i++) {
//            if(s.charAt(i) != s.charAt(s.length()-i-1)) return false;
//        }
//        return true;
        if(x<0 || x%10==0 && x!=0) return false;
        int r = 0;
        while(x > r) {
            r = r*10 + x%10;
            x /= 10;
        }
        return x==r || x==r/10;
    }

    //13
    public int romanToInt(String s) {
//        Map<Character,Integer> map = new HashMap<>();
//        map.put('I', 1);
//        map.put('V', 5);
//        map.put('X', 10);
//        map.put('L', 50);
//        map.put('C', 100);
//        map.put('D', 500);
//        map.put('M', 1000);
        int i = 0, res = 0;
        while (i < s.length()) {
            char c = '0';
            if (i + 1 < s.length()) {
                c = s.charAt(i + 1);
            }
            switch (s.charAt(i)) {
                case 'I':
                    if (c == 'V' || c == 'X') {
                        res += c == 'V' ? 4 : 9;
                        i++;
                    } else res += 1;
                    break;
                case 'V':
                    res += 5;
                    break;
                case 'X':
                    if (c == 'L' || c == 'C') {
                        res += c == 'L' ? 40 : 90;
                        i++;
                    } else res += 10;
                    break;
                case 'L':
                    res += 50;
                    break;
                case 'C':
                    if (c == 'D' || c == 'M') {
                        res += c == 'D' ? 400 : 900;
                        i++;
                    } else res += 100;
                    break;
                case 'D':
                    res += 500;
                    break;
                case 'M':
                    res += 1000;
                    break;
//                default:
//                    res += map.get(s.charAt(i));
            }
            i++;
        }
        return res;
    }

    //14
    public String longestCommonPrefix(String[] strs) {
//        StringBuilder res = new StringBuilder();
//        if(strs.length==0) return res.toString();
//        int i=0;
//        while(true) {
//            char c;
//            if(i<strs[0].length()) c = strs[0].charAt(i);
//            else break;
//            int j;
//            for(j=1;j<strs.length;j++) {
//                if(i>=strs[j].length() || strs[j].charAt(i)!=c) break;
//            }
//            if(j<strs.length) break;
//            res.append(c);
//            i++;
//        }
//        return res.toString();

        if(strs.length==0) return "";
        String res = strs[0];
        for(int i=1;i<strs.length;i++) {
            int j=0;
            for(;j<strs[i].length()&&j<res.length();j++) {
                if(res.charAt(j) != strs[i].charAt(j)) break;
            }
            res = res.substring(0,j);
            if(res.equals("")) return res;
        }
        return res;
    }

    //124
    int maxSum = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        maxGain(root);
        return maxSum;
    }

    private int maxGain(TreeNode root) {
        if(root == null) return 0;
        int l = Math.max(maxGain(root.left),0);
        int r = Math.max(maxGain(root.right),0);
        int sum = l + r + root.val;
        maxSum = Math.max(sum,maxSum);
        return Math.max(l,r) + root.val;
    }

    //125
    public boolean isPalindrome(String s) {
        int i=0, j=s.length()-1;
        while(i<j) {
            while(i<j && !Character.isLetterOrDigit(s.charAt(i))) {
                i++;
            }
            while(i<j && !Character.isLetterOrDigit(s.charAt(j))) {
                j--;
            }
            if(Character.toLowerCase(s.charAt(i++)) != Character.toLowerCase(s.charAt(j--))) {
                return false;
            }
        }
        return true;
    }

    //1028
    int index=0, n=0;
    public TreeNode recoverFromPreorder(String S) {
//        TreeNode root = new TreeNode(S.charAt(0)-'0');
//        Stack<TreeNode> stack = new Stack<>();
//        TreeNode p = root;
//        int i = 1,cnt=0,t=0;
//        while(i<S.length()) {
//            while(S.charAt(i)=='-') {
//                cnt++;
//                i++;
//            }
//
//        }
        TreeNode root = recoverFromPreorder(0,S);
        return root;
    }
    private TreeNode recoverFromPreorder(int cnt, String s) {
        int num = 0;
        while(index<s.length() && s.charAt(index)>='0' && s.charAt(index)<='9') {
            num = num*10 + s.charAt(index++)-'0';
        }
        TreeNode root = new TreeNode(num);
        if(index >= s.length()) return root;
        n = 0;
        while(s.charAt(index) == '-') {
            n++;
            index++;
        }
        if(n==cnt+1) {
            root.left = recoverFromPreorder(n,s);
        }
        if(index >= s.length()) return root;
        if(n==cnt+1) {
            root.right = recoverFromPreorder(n,s);
        }
        return root;
    }
}
