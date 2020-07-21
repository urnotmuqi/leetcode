package com.muqi.leetcode;

import java.util.*;
import java.util.concurrent.locks.Condition;

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

    //4
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int l1 = nums1.length, l2 = nums2.length;
        if((l1+l2)%2==1) {
            return getKthElement(nums1, nums2 , (l1+l2)/2+1);
        } else {
            return (getKthElement(nums1, nums2, (l1+l2)/2) + getKthElement(nums1, nums2, (l1+l2)/2+1)) / 2.0;
        }

    }
    private int getKthElement(int[] nums1, int[] nums2, int k) {
        int l1 = nums1.length, l2 = nums2.length;
        int i=0,j=0;
        while(true) {
            if(i == l1) return nums2[j+k-1];
            if(j == l2) return nums1[i+k-1];
            if(k == 1) return Math.min(nums1[i], nums2[j]);
            int index1 = Math.min(i+k/2,l1)-1;
            int index2 = Math.min(j+k/2,l2)-1;
            if(nums1[index1] <= nums2[index2]) {
                k -= (index1-i+1);
                i = index1+1;
            }
            else {
                k -= (index2-j+1);
                j = index2+1;
            }
        }
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

    //32
    public int longestValidParentheses(String s) {
//        // 栈
//        Stack<Integer> stack = new Stack<>();
//        int maxl=0;
//        stack.push(-1);
//        for(int i=0;i<s.length();i++) {
//            char t = s.charAt(i);
//            if(t == '(') stack.push(i);
//            else {
//                stack.pop();
//                if(!stack.isEmpty()) {
//                    maxl = Math.max(maxl, i-stack.peek());
//                }
//                else {
//                    stack.push(i);
//                }
//            }
//        }
//        return maxl;

//        //动态规划
//        int[] dp = new int[s.length()];
//        int maxl=0;
//        for(int i=1;i<s.length();i++) {
//            if(s.charAt(i) == ')') {
//                if (s.charAt(i - 1) == '(') {
//                    if (i - 2 > 0) {
//                        dp[i] = dp[i - 2];
//                    }
//                    dp[i] += 2;
//                } else if (i - dp[i - 1] > 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
//                    if (i - dp[i - 1] - 2 > 0) dp[i] = dp[i - dp[i - 1] - 2];
//                    dp[i] += dp[i - 1] + 2;
//                }
//                maxl = Math.max(dp[i], maxl);
//            }
//        }
//        return maxl;

        //贪心
        int left=0, right=0, maxl=0;
        for(int i=0;i<s.length();i++) {
            if(s.charAt(i) == '(') left++;
            else right++;
            if(left == right) {
                maxl = Math.max(maxl, right*2);
            }
            else if(right > left) {
                left = 0;
                right = 0;
            }
        }
        left=0;
        right=0;
        for(int i=s.length()-1;i>=0;i--) {
            if((s.charAt(i) == '(')) left++;
            else right++;
            if(left == right) maxl = Math.max(maxl, left*2);
            else if(left > right) {
                left = 0;
                right = 0;
            }
        }
        return maxl;
    }

    //44
    public boolean isMatch(String s, String p) {
        char[] chs = s.toCharArray();
        char[] chp = p.toCharArray();
        boolean[][] dp = new boolean[s.length()+1][p.length()+1];
        for(int i=0;i<=chs.length;i++) {
            for(int j=0;j<=chp.length;j++) {
                if(j==0) dp[i][j] = i==0 ;
                else if(chp[j-1] == '*') {
                    dp[i][j] = dp[i][j-1];
                    if(i>0) {
                        dp[i][j] |= dp[i-1][j];
                    }
                }
                else {
                    if(i>0 && (chp[j-1] == chs[i-1] || chp[j-1]=='?')) {
                        dp[i][j] = dp[i-1][j-1];
                    }
                }
            }
        }
        return dp[chs.length][chp.length];
    }

    //63
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int[] dp = new int[obstacleGrid[0].length];
        dp[0] = 1;
        for(int i=0;i<obstacleGrid.length;i++) {
            for(int j=0;j<obstacleGrid[0].length;j++) {
                if(obstacleGrid[i][j] == 1) {
                    dp[j] = 0;
                }
                else if(j-1>=0) {
                    dp[j] += dp[j-1];
                }
            }
        }
        return dp[obstacleGrid[0].length-1];
    }

    //67
    public String addBinary(String a, String b) {
        int f=0;
        StringBuilder res = new StringBuilder();
        for(int i=a.length()-1,j=b.length()-1;i>=0 || j>=0;i--,j--) {
            int sum = f;
            sum += i>=0 ? a.charAt(i)-'0' : 0;
            sum += j>=0 ? b.charAt(j)-'0' : 0;
            res.append(sum%2);
            f = sum/2;
        }
        if(f==1) res.append(f);
        return res.reverse().toString();
    }

    //97
    public boolean isInterleave(String s1, String s2, String s3) {
        if(s1.length()+s2.length() != s3.length()) return false;
        return isInterleave(s1,s2,s3,0,0,0);
    }
    private boolean isInterleave(String s1, String s2, String s3, int i1, int i2, int i3) {
        if(i1==s1.length() && i2==s2.length() && i3==s3.length()) return true;
        else if(i3==s3.length()) return false;
        if(i1<s1.length() && s3.charAt(i3)==s1.charAt(i1)) {
            if (isInterleave(s1, s2, s3, i1 + 1, i2, i3 + 1)) return true;
        }
        if(i2<s2.length() && s3.charAt(i3)==s2.charAt(i2)) return isInterleave(s1,s2,s3,i1,i2+1,i3+1);
        return false;
    }

    //108
    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBST(nums, 0, nums.length-1);
    }
    private TreeNode sortedArrayToBST(int[] nums, int start, int end) {
        if(start>end) return null;
        int len = (end-start+1)/2;
        TreeNode root = new TreeNode(nums[start+len]);
        root.left = sortedArrayToBST(nums, start, start+len-1);
        root.right = sortedArrayToBST(nums, start+len+1, end);
        return root;
    }

    public int minDepth(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int h=1;
        while(!queue.isEmpty()) {
            int n = queue.size();
            for(int i=0;i<n;i++) {
                TreeNode p = queue.poll();
                if(p.left==null&&p.right==null) return h;
                if(p.left!=null) queue.add(p.left);
                if(p.right!=null) queue.add(p.right);
            }
            h++;
        }
        return h;
    }

    //112
    public boolean hasPathSum(TreeNode root, int sum) {
        if(root == null) return false;
        sum -= root.val;
        if(root.left==null && root.right==null && sum==0) return true;
        if(hasPathSum(root.left, sum)) return true;
        if(hasPathSum(root.right, sum)) return true;
        return false;
    }

    //120
    public int minimumTotal(List<List<Integer>> triangle) {
        int m = triangle.size();
        int n = triangle.get(m-1).size();
        int[] dp = new int[n+1];
        for(int i=m-1;i>=0;i--) {
            for(int j=triangle.get(i).size()-1;j>=0;j--) {
                dp[j] = triangle.get(i).get(j) + Math.min(dp[j], dp[j+1]);
            }
        }
        return dp[0];
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

    //209
    public int minSubArrayLen(int s, int[] nums) {
        int i=0, minl=0, sum=0, l=0;
        for(int j=0;j<nums.length;j++) {
            sum += nums[j];
            l++;
            while(sum>=s) {
                if(minl>l || minl==0) minl=l;
                sum -= nums[i++];
                l--;
            }
        }
        return minl;
    }

    //215
    //  快速排序
//    public int findKthLargest(int[] nums, int k) {
//        return quick(nums,k, 0, nums.length-1);
//    }
//    private int quick(int[] nums, int k, int start, int end) {
//        Random random = new Random();
//        int t = random.nextInt(end-start+1)+start;
//        int temp = nums[t];
//        nums[t] = nums[start];
//        int i = start, j = end;
//        while(i<j) {
//            while(temp>=nums[j] && i<j) {
//                j--;
//            }
//            if(i<j) nums[i++] = nums[j];
//            while(temp<=nums[i] && i<j) {
//                i++;
//            }
//            if(i<j) nums[j--] = nums[i];
//        }
//        nums[i] = temp;
//        if(i==k-1) return temp;
//        else if(i<k-1) return quick(nums, k, i+1, end);
//        else return quick(nums, k, start, i-1);
//    }

    //堆排序
    public int findKthLargest(int[] nums, int k) {
        buildMaxHeap(nums);
        int len = nums.length-1;
        for(int i=nums.length-1;i>nums.length-k;i--) {
            nums[0] = nums[i];
            adjustHeap(nums, 0, len--);
        }
        return nums[0];
    }

    private void buildMaxHeap(int[] nums) {
        int len = 1;
        for(int i=nums.length-1;i>=0;i--) {
            adjustHeap(nums, i, len++);
        }
    }

    private void adjustHeap(int[] nums, int i, int len) {
        int p,child, t=nums[i];
        for(p=i;p*2+1<i+len;p=child) {
            if(p*2+2<i+len && nums[p*2+2]>nums[p*2+1]) child = p*2+2;
            else child = p*2+1;
            if(nums[child]>t) {
                nums[p] = nums[child];
            }
            else break;
        }
        nums[p] = t;
    }

    //224
    int i=0;
    public int calculate(String s) {
        Stack<Integer> stack = new Stack<>();
        char[] chs = s.toCharArray();
        char sign = '+';
        int num = 0;
        while(i<chs.length) {
            char x = chs[i++];
            if(Character.isDigit(x)) {
                num = num*10 + (x-'0');
            }
            if(x == '(') {
                num = calculate(s);
            }
            if((x !=' ' && !Character.isDigit(x)) || i==chs.length){
                int pre;
                switch(sign) {
                    case '+' :
                        stack.push(num);
                        break;
                    case '-' :
                        stack.push(-num);
                        break;
                }
                sign = x;
                num = 0;
            }
            if(x == ')') break;
        }
        int res = 0;
        while(!stack.isEmpty()) {
            res += stack.pop();
        }
        return res;
    }

    //227
    public int calculate1(String s) {
        Stack<Integer> s1 = new Stack<>();
        Stack<Character> s2 = new Stack<>();
        int res = 0;
        for(int i=0;i<s.length();i++) {
            char x = s.charAt(i);
            if(x ==' ') continue;
            else if(Character.isDigit(x)) {
                int num2 = x-'0';
                i++;
                while(i<s.length() && Character.isDigit(s.charAt(i))) {
                    char ch = s.charAt(i);
                    num2 = num2*10 + ch-'0';
                    i++;
                }
                i--;
                s1.push(num2);
            }
            else if(x == '+' || x == '-') {
                while(!s2.isEmpty()) {
                    char c = s2.pop();
                    int num2 = s1.pop();
                    int num1 = s1.pop();
                    s1.push(cal1(num1, num2, c));
                }
                s2.push(x);
            }
            else {
                if(!s2.isEmpty() && (s2.peek()=='*' || s2.peek() =='/')) {
                    char c = s2.pop();
                    int num2 = s1.pop();
                    int num1 = s1.pop();
                    s1.push(cal1(num1, num2, c));
                }
                s2.push(x);
            }
        }
        while(!s2.isEmpty()) {
            char c = s2.pop();
            int num2 = s1.pop();
            int num1 = s1.pop();
            s1.push(cal1(num1, num2, c));
        }
        return s1.pop();
    }

    private int cal1(int a, int b, char c) {
        int res = 0;
        switch(c) {
            case '+' :
                res = a+b;
                break;
            case '-' :
                res = a-b;
                break;
            case '*' :
                res = a*b;
                break;
            case '/' :
                res = a/b;
                break;
        }
        return res;
    }

    //315
    public List<Integer> countSmaller(int[] nums) {
        List<Integer> list = new ArrayList<>();
        Set<Integer> set = new TreeSet<>();
        for(int num : nums) {
            set.add(num);
        }
        Map<Integer, Integer> map = new HashMap<>();
        int rank = 1;
        for(int num : set) {
            map.put(num, rank++);
        }

        FenwickTree fenwickTree = new FenwickTree(set.size()+1);
        for(int i=nums.length-1;i>=0;i--) {
            rank = map.get(nums[i]);
            fenwickTree.update(rank, 1);
            list.add(fenwickTree.query(rank-1));
        }
        Collections.reverse(list);
        return list;
    }

    private class FenwickTree {
        private int[] tree;
        private int len;

        public FenwickTree(int len) {
            this.len = len;
            tree = new int[len+1];
        }

        public void update(int i, int delta) {
            while(i<=this.len) {
                tree[i] += delta;
                i += lowbit(i);
            }
        }

        public int query(int i) {
            int sum = 0;
            while(i>0) {
                sum += tree[i];
                i -= lowbit(i);
            }
            return sum;
        }

        public int lowbit(int x) {
            return x & (-x);
        }
    }

    //350
    public int[] intersect(int[] nums1, int[] nums2) {
        Set<Integer> set = new HashSet<>();
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int i=0,j=0,index=0;
        while(i<nums1.length && j<nums2.length) {
            if(nums1[i] == nums2[j]) {
                set.add(nums1[i]);
                j++;
            }
            else if(nums1[i]<nums2[j]) {
                i++;
            }
            else j++;
        }
        int[] res = new int[set.size()];
        for(int num:set) {
            res[index++] = num;
        }
        return res;
    }

    //378
    public int kthSmallest(int[][] matrix, int k) {
        int len = matrix.length, left = matrix[0][0];
        int right = matrix[len-1][len-1];
        while(left < right) {
            int mid = (left + right)/2;
            if(check(matrix, mid, k, len)) {
                right = mid;
            }
            else left = mid + 1;
        }
        return left;
    }

    private boolean check(int[][] a, int mid, int k, int len) {
        int i=len-1, j=0, n=0;
        while(i>=0 && j<len) {
            if(a[i][j]<=mid) {
                n += i+1;
                j++;
            }
            else i--;
        }
        return n>=k;
    }

    //718
//    public int findLength(int[] A, int[] B) {
//        //动态规划
////        int maxl = 0;
////        int[] dp = new int[B.length+1];
////        for(int i=1;i<=A.length;i++) {
////            for(int j=B.length;j>=1;j--) {
////                if(A[i-1] == B[j-1]) {
////                    dp[j] = dp[j-1] + 1;
////                }
////                else dp[j] = 0;
////                maxl = Math.max(maxl, dp[j]);
////            }
////        }
////        return maxl;
//
//        //滑动窗口
//        int maxl=0, len1=A.length, len2=B.length;
//        for(int i=0;i<A.length;i++) {
//            int len = Math.min(len2, len1-i);
//            maxl = Math.max(maxl, maxLength(A, B, i, 0, len));
//        }
//        for(int i=0;i<B.length;i++) {
//            int len = Math.min(len1, len2-i);
//            maxl = Math.max(maxl, maxLength(A, B, 0, i, len));
//        }
//        return maxl;
//    }
//    private int maxLength(int a[], int b[], int start1, int start2, int len) {
//        int maxl=0, k=0;
//        for(int i=0;i<len;i++) {
//            if(a[start1+i] == b[start2+i]) {
//                k++;
//            }
//            else k=0;
//            maxl = Math.max(maxl, k);
//        }
//        return maxl;
//    }

    //二分+哈希
    public int findLength(int[] A, int[] B) {
        int left=1, right=Math.min(A.length,B.length)+1;
        while(left<right) {
            int mid = (left+right)/2;
            if(check(A, B, mid)) left = mid+1;
            else right=mid;
        }
        return left-1;
    }
    private boolean check(int[] a, int[] b, int len) {
        int mod = 1000000009, base = 113;
        long hash = 0;
        for(int i=0;i<len;i++) {
            hash = (hash*base + a[i])%mod;
        }
        Set<Long> set = new HashSet<>();
        set.add(hash);
        long mult = qPow(base, len-1);
        for(int i=len;i<a.length;i++) {
            hash = ((hash - a[i - len] * mult % mod + mod) % mod * base + a[i]) % mod;
            set.add(hash);
        }
        hash = 0;
        for(int i=0;i<len;i++) {
            hash = (hash*base + b[i])%mod;
        }
        if(set.contains(hash)) return true;
        for(int i=len;i<b.length;i++) {
            //hash = ((hash - b[i-len]*mult%mod)*base + b[i])%mod;    不能用这个是因为前面取余过，相减之后可能会是负数
            hash = ((hash - b[i - len] * mult % mod + mod) % mod * base + b[i]) % mod;
            if(set.contains(hash)) return true;
        }
        return false;
    }
    private long qPow(long x, long n) {
        int mod = 1000000009;
        long res = 1;
        while(n!=0) {
            if((n&1)==1) {
                res = res*x%mod;
            }
            x = x*x%mod;
            n /= 2;
        }
        return res;
    }

    //752
    public int openLock(String[] deadends, String target) {
        Set<String> set = new HashSet<>();
        for(int i=0;i<deadends.length;i++) {
            set.add(deadends[i]);
        }
        if(set.contains("0000")) return -1;
        Set<String> q1 = new HashSet<>();
        Set<String> q2 = new HashSet<>();
        q1.add("0000");
        q2.add(target);
        set.add("0000");
        set.add(target);
        int min = 1;
        while(!q1.isEmpty() && !q2.isEmpty()) {
            Set<String> temp = new HashSet<>();
            if(q2.size()<q1.size()) {
                temp = q1;
                q1=q2;
                q2=temp;
            }
            temp.clear();
            for(String str : q1) {
                for(int j=0;j<4;j++) {
                    String t = up(str, j);
                    if(q2.contains(t)) return min;
                    if(!set.contains(t)) {
                        temp.add(t);
                        set.add(t);
                    }
                    t = down(str, j);
                    if(q2.contains(t)) return min;
                    if(!set.contains(t)) {
                        temp.add(t);
                        set.add(t);
                    }
                }
            }
            q1=temp;
            min++;
        }
        return -1;
    }

    private String up(String str, int i) {
        char[] chs = str.toCharArray();
        if(chs[i]=='9') chs[i]='0';
        else chs[i] += 1;
        return new String(chs);
    }

    private String down(String str, int i) {
        char[] chs= str.toCharArray();
        if(chs[i]=='0') chs[i] = '9';
        else chs[i] -= 1;
        return new String(chs);
    }

    //785
    public boolean isBipartite(int[][] graph) {
        int n = graph.length;
        int[] color = new int[n];
        Arrays.fill(color, 0);
        for(int i=0;i<n;i++) {
            if(color[i]==0) {
                if(!dfs(color, i, graph)) return false;
            }
        }
        return true;
    }

    private boolean dfs(int[] color, int v, int[][] graph) {
        int c = color[v]==1 ? 2 : 1;
        for(int i=0;i<graph[v].length;i++){
            if(color[graph[v][i]]==0) {
                color[graph[v][i]] = c;
                if(!dfs(color, graph[v][i], graph)) return false;
            }
            else if(color[graph[v][i]]==color[v]) {
                return false;
            }
        }
        return true;
    }

    //887
    public int superEggDrop(int K, int N) {
        if(K==1) return N;
        if(N==0) return 0;
        int res = Integer.MAX_VALUE;
        for(int i=1;i<=N;i++) {
            res = Math.min(res, Math.max(superEggDrop(K, N-i), superEggDrop(K-1,i-1))+1);
        }
        return res;
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
