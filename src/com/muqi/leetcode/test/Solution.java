package com.muqi.leetcode.test;


import com.sun.org.apache.xpath.internal.operations.Bool;
import sun.reflect.generics.tree.Tree;

import java.util.*;

/**
 * @author muqi
 * @since 2020/4/28 18:41
 */
public class Solution {
    //面试题03
    public int findRepeatNumber(int[] nums) {
        Map<Integer,Integer> map = new HashMap<>();
        int i;
        for(i=0;i<nums.length;i++) {
            if(map.containsKey(nums[i])) break;
            else map.put(nums[i],1);
        }
        return nums[i];
    }

    //面试题04
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        int n = matrix.length;
        if(n == 0) return false;
        int j = matrix[0].length - 1;
        int i = 0;
        while(i < n && j > 0) {
            if (matrix[i][j] > target) j--;
            else if (matrix[i][j] < target) i++;
            else return true;
        }
        return false;
    }

    //面试题05
    public String replaceSpace(String s) {
        String s1 = " ";
        String s2 = "%20";
        s = s.replaceAll(s1,s2);
        return s;
    }

    //面试题06
    public int[] reversePrint(ListNode head) {
        ListNode l = head;
        int length = 0;
        while(l != null) {
            length ++;
            l = l.next;
        }
        int[] result = new int[length];
        for(int i=length-1;i>=0;i--) {
            result[i] = head.val;
            head = head.next;
        }
        return result;
    }

    static int r;

    //面试题07
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        Map<Integer,Integer> map = new HashMap<>();
        for(int i=0;i<inorder.length;i++) {
            map.put(inorder[i],i);
        }
        if(inorder.length == 0) return null;
        r=0;
        TreeNode root = build(preorder,inorder,0,preorder.length-1,map);
        return root;
    }

    private TreeNode build(int[] preorder, int[] inorder, int left, int right, Map<Integer,Integer> map) {
        TreeNode root = new TreeNode(preorder[r]);
        int t = map.get(preorder[r]);
        r++;
        if(left < t) root.left = build(preorder,inorder,left,t-1,map);
        if(right > t) root.right = build(preorder,inorder,t+1,right,map);
        return root;
    }

    //面试题10-I
    public int fib(int n) {
        if(n == 0 || n == 0) return n;
        int a = 1;
        int b = 0;
        for(int i=1;i<n;i++) {
            a = a + b;
            b = a - b;
            a %= 1000000007;
        }
        return a;
    }

    //面试题10-II
    public int numWays(int n) {
        if(n == 0 || n == 1) return 1;
        int a = 1;
        int b = 1;
        for(int i=1;i<n;i++) {
            a = a + b;
            b = a - b;
            a %= 1000000007;
        }
        return a;
    }

    //面试题11
    public int minArray(int[] numbers) {
        for(int i=0;i<numbers.length-1;i++) {
            if(numbers[i] > numbers[i+1]) {
                return numbers[i+1];
            }
        }
        return numbers[0];
    }

    //面试题12
    public boolean exist(char[][] board, String word) {
        char[] chs = word.toCharArray();
        boolean[][] flag = new boolean[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == chs[0]) {
                    if(find(flag, board,chs,0,i,j)) return true;
                }
            }
        }
        return false;
    }
    private boolean find(boolean[][] flag, char[][] board, char[] chs, int k, int p, int q) {
        if(k == chs.length) return true;
        if(p<0 || q<0 || p>=board.length || q>= board[0].length || flag[p][q] || board[p][q] != chs[k]) return false;
        flag[p][q] = true;
        boolean res = find(flag, board, chs, k+1, p, q+1 ) || find(flag, board,chs,k+1,p+1,q)
                || find(flag,board,chs,k+1,p,q-1) || find(flag,board,chs,k+1,p-1,q);
        flag[p][q] = false;
        return res;
    }

    //面试题13
    public int movingCount(int m, int n, int k) {
        boolean[][] flag = new boolean[m][n];
        return dfs(flag,m,n,0,0,k,0);
    }

    public int dfs(boolean[][] flag,int m,int n,int i,int j,int k,int cnt) {
        if(i>=m || j>=n || flag[i][j] || (i/10 + i%10 + j/10 + j%10) > k)
            return cnt;
        cnt++;
        flag[i][j] = true;
        cnt = dfs(flag,m,n,i,j+1,k,cnt);
        cnt = dfs(flag,m,n,i+1,j,k,cnt);
        return cnt;
    }

    //面试题14 剪绳子
    /**
     * 剪成长度2或3
     * 因为当x>=5时  3*(x-3)>x   就是把x分成3和另两段时，乘积会大于x
     *    当x=4时  x=2*2
     */
    public int cuttingRope(int n) {
        if(n<=2) return 1;
        if(n == 3) return 2;
        if(n == 4) return 4;
        if(n%3 == 0) {
            int cnt = n/3;
            return (int)cal(cnt);
        }
        if(n%3 == 1) {
            int cnt = n/3-1;
            return (int)(cal(cnt)*2*2%1000000007);
        }
        int cnt = n/3;
        return (int)cal(cnt)*2%1000000007;
    }

    private long cal(int n) {
        long sum =1;
        for(int i=0;i<n;i++) {
            sum *= 3;
            if(sum > 1000000007) sum %= 1000000007;
        }
        return sum;
    }

    //面试题15

    /**
     * 还可用n&(n-1） 把n最右边的1变为0
     * @param n
     * @return
     */
    public int hammingWeight(int n) {
        int cnt = 0;
        while(n != 0) {
            cnt = n&1;    //按位与，当n最右边为1时，n&1=1
            n >>>= 1;     //无符号右移，左边补0；>>为带符号位移，整数补0，负数补1
        }
        return cnt;
    }

    //面试题16   用快速幂，时间复杂度为O(log2 n )
    public double myPow(double x, int n) {
        if(n == -1) return 1/x;
        if(n == 0) return 1;
        if(n == 1) return x;
        double half = myPow(x,n/2);
        double mod = myPow(x,n%2);
        return half*half*mod;
    }

    //面试题17
    public int[] printNumbers(int n) {
        int cnt = (int)Math.pow(10,n)-1;
        int[] nums = new int[cnt];
        for(int i=0;i<cnt;i++) {
            nums[i] = i + 1;
        }
        return nums;
    }

    //面试题18
    public ListNode deleteNode(ListNode head, int val) {
        if(head.val == val) return head.next;
        ListNode p = head;
        ListNode q = head.next;
        while(q != null) {
            if(q.val == val) {
                p.next = q.next;
                break;
            }
            p = p.next;
            q = q.next;
        }
        return head;
    }

    //19
    public boolean isMatch(String s, String p) {
        char[] chs = s.toCharArray();
        char[] chp = p.toCharArray();
        boolean[][] f = new boolean[chs.length+1][chp.length+1];   //+1是为了方便对空串的处理
        for(int i=0;i<=chs.length;i++) {
            for(int j=0;j<=chp.length;j++) {
                if(j==0) {
                    f[i][j] = i==0;
                }
                else {
                    if(chp[j-1]!='*') {
                        if(i>0 && (chs[i-1] == chp[j-1] || chp[j-1] == '.')) {
                            f[i][j] = f[i-1][j-1];
                        }
                    }
                    else {
                        if(j>=2) {
                            f[i][j] = f[i][j-2];
                        }
                        if(i>=1 && j>=2 && (chs[i-1] == chp[j-2] || chp[j-2] == '.')) {
                            f[i][j] |= f[i-1][j];
                        }
                    }
                }
            }
        }
        return f[chs.length][chp.length];
    }


    //20
    public boolean isNumber(String s) {
        char[] str = s.toCharArray();
        int i = 0;
        boolean point = false;
        boolean haveE = false;
        while(i<str.length && str[i] == ' ') {
            i++;
        }
        if(i<str.length && (str[i] == '+' || str[i] == '-' || str[i] == '.')) {
            if(str[i] == '.') point = true;
            i++;
        }
        if(i<str.length && (Character.isDigit(str[i]) || str[i] == '.')) {
            while(i < str.length) {
                while(Character.isDigit(str[i])) {
                    i++;
                    if(i == str.length) return true;
                }
                if(str[i] == '.' && !point && !haveE) {
                    point = true;
                    i++;
                    if(i==str.length && !Character.isDigit(str[i-2])) return false;
                }
                else if(str[i] == 'e' && !haveE ) {
                    haveE = true;
                    i++;
                    if(i<str.length && (str[i] == '-' || str[i] == '+')) i++;
                    if(i==str.length || !Character.isDigit(str[i])) return false;
                }
                else if (str[i] == ' ') {
                    while(i<str.length && str[i] == ' ') {
                        i++;
                    }
                    if(i != str.length) return false;
                }
                else return false;
            }
            return true;
        }
        return false;
    }

    //21
    public int[] exchange(int[] nums) {
        int p = 0;
        int q = nums.length-1;
        while(p < q) {
            while(p<nums.length && nums[p] % 2 == 1) {
                p++;
            }
            while(q>=0 && nums[q] % 2 == 0) {
                q--;
            }
            if(p<q) {
                int t = nums[p];
                nums[p] = nums[q];
                nums[q] = t;
            }
        }
        return nums;
    }

    //22
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode p = head;
        for(int i=0;i<k;i++) {
            p = p.next;
        }
        while(p != null) {
            p = p.next;
            head = head.next;
        }
        return head;
    }

    //24
    public ListNode reverseList(ListNode head) {
        ListNode p = null;
        while(head != null) {
            ListNode q = head;
            head = head.next;
            q.next = p;
            p = q;
        }
        return p;
    }

    //25
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode head = new ListNode(0);
        ListNode p = head;
        while(l1 != null && l2 != null) {
            if(l1.val < l2.val) {
                p.next = l1;
                p = p.next;
                l1 = l1.next;
            }
            else {
                p.next = l2;
                p = p.next;
                l2 = l2.next;
            }
        }
        if(l1 != null) {
            p.next = l1;
        }
        if(l2 != null) {
            p.next = l2;
        }
        return head.next;
    }

    //26
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if(A == null || B == null) return false;
        return isSameStructure(A,B) || isSubStructure(A.left,B) || isSubStructure(A.right,B);
    }

    private boolean isSameStructure(TreeNode A, TreeNode B) {
        if(B==null) return true;
        if(A==null) return false;
        return A.val == B.val && isSameStructure(A.left, B.left) && isSameStructure(A.right,B.right);
    }

    //27
    public TreeNode mirrorTree(TreeNode root) {
        if(root == null) return root;
        TreeNode p = root.left;
        root.left = mirrorTree(root.right);
        root.right = mirrorTree(p);
        return root;
    }

    //28
    public boolean isSymmetric(TreeNode root) {
        if(root == null) return true;
        return isSameStructure2(root.left, root.right);
    }

    private boolean isSameStructure2(TreeNode A, TreeNode B) {
        if(B==null && A==null) return true;
        if(A==null || B==null) return false;
        return A.val == B.val && isSameStructure2(A.left, B.right) && isSameStructure2(A.right,B.left);
    }

    //29
    public int[] spiralOrder(int[][] matrix) {
        if(matrix.length == 0) return new int[0];
        int[] result = new int[matrix.length*matrix[0].length];
        int l=0;
        int t=0;
        int r=matrix[0].length-1;
        int b=matrix.length-1;
        int k=0;
        while(true) {
            for(int i=l;i<=r;i++) {
                result[k] = matrix[t][i];
                k++;
            }
            t++;
            if(t>b) break;
            for(int i=t;i<=b;i++) {
                result[k] = matrix[i][r];
                k++;
            }
            r--;
            if(r<l) break;
            for(int i=r;i>=l;i--) {
                result[k] = matrix[b][i];
                k++;
            }
            b--;
            if(b<t) break;
            for(int i=b;i>=t;i--) {
                result[k] = matrix[i][l];
                k++;
            }
            l++;
            if(l>r) break;
        }
        return result;
    }

    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Stack<Integer> s = new Stack<>();
        int i=0;
        int j=0;
        while(i<pushed.length && j<popped.length) {
            if(!s.empty() && popped[j]==s.peek()) {
                s.pop();
                j++;
            }
            else {
                s.push(pushed[i]);
                i++;
            }
        }
        if(i==j) return true;
        return false;
    }

    //32-1
//    public int[] levelOrder(TreeNode root) {
//        if(root == null) return int[0];
//        Queue<TreeNode> q = new LinkedList<>();
//        ArrayList<Integer> list = new ArrayList<>();
//        q.add(root);
//        while(!q.isEmpty()) {
//            TreeNode p = q.poll();
//            list.add(p.val);
//            if(p.left != null) q.add(p.left);
//            if(p.right != null) q.add(p.right);
//        }
//        int[] res = new int[list.size()];
//        for(int i=0;i<list.size();i++) {
//            res[i] = list.get(i);
//        }
//        return res;
//    }

    //32-2
//    public List<List<Integer>> levelOrder(TreeNode root) {
//        Queue<TreeNode> q = new LinkedList<>();
//        List<List<Integer>> res = new ArrayList<>();
//        List<Integer> list = new ArrayList<>();
//        q.add(null);
//        if(root != null) q.add(root);
//        while(!q.isEmpty()) {
//            TreeNode p = q.poll();
//            if(p == null && !q.isEmpty()) {
//                list = new ArrayList<>();
//                res.add(list);
//                q.add(null);
//            }
//            else if(p != null){
//                list.add(p.val);
//                if(p.left != null) q.add(p.left);
//                if(p.right != null) q.add(p.right);
//            }
//        }
//        return res;
//    }

    //32-3
    public List<List<Integer>> levelOrder(TreeNode root) {
        Queue<TreeNode> q = new LinkedList<>();
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        int cnt = 0;
        q.add(null);
        if(root != null) q.add(root);
        while(!q.isEmpty()) {
            TreeNode p = q.poll();
            if(p == null && !q.isEmpty()) {
                if(cnt>0 && cnt%2 == 0) Collections.reverse(list);
                cnt++;
                list = new ArrayList<>();
                res.add(list);
                q.add(null);
            }
            else if(p != null){
                list.add(p.val);
                if(p.left != null) q.add(p.left);
                if(p.right != null) q.add(p.right);
            }
        }
        return res;
    }

    //33
    public boolean verifyPostorder(int[] postorder) {
        return isCorrect(postorder,0,postorder.length-1);
    }

    private boolean isCorrect(int[] postorder, int l, int r) {
        if(l>=r) return true;
        int root = postorder[r];
        int i=l;
        while(i<r && postorder[i] < root) {
            i++;
        }
        int j=i-1;
        while(i<r && postorder[i] > root) {
            i++;
        }
        if(i == r) return isCorrect(postorder,l,j) && isCorrect(postorder,j+1,r-1);
        return false;
    }

    //34
    List<List<Integer>> res = new LinkedList<>();
    LinkedList<Integer> list = new LinkedList<>();
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        preorder(root, sum);
        return res;
    }

    private void preorder(TreeNode root, int sum) {
        if(root == null) return;
        sum -= root.val;
        list.add(root.val);
        if(sum == 0 && root.left == null && root.right == null) {
            res.add(new LinkedList<>(list));
        }
        preorder(root.left, sum);
        preorder(root.right, sum);
        list.removeLast();
    }

    //35
//    public Node copyRandomList(Node head) {
//        Node root = new Node(0);
//        Node t = head;
//        Node p = root;
//        Map<Node,Node> map = new HashMap<>();
//        while(t != null) {
//            Node q = new Node(t.val);
//            p.next = q;
//            map.put(t,q);
//            t = t.next;
//            p = q;
//        }
//        while(head != null) {
//            if(head.random == null) map.get(head).random = null;
//            else map.get(head).random = map.get(head.random);
//            head = head.next;
//        }
//        return root.next;
//    }

    //36
    Node pre,head;
    public Node treeToDoublyList(Node root) {
        if(root==null) return null;
        inorder(root);
        pre.right = head;
        head.left = pre;
        return head;
    }

    private void inorder(Node root) {
        if(root == null) return;
        inorder(root.left);
        if(head == null) head = root;
        if(pre!=null) pre.right = root;
        root.left = pre;
        pre = root;
        inorder(root.right);
    }

    //38
    LinkedList<String> list1 = new LinkedList<>();
    char[] c;
    public String[] permutation(String s) {
        c = s.toCharArray();
        dfs(0);
        return list1.toArray(new String[list1.size()]);
    }

    private void dfs(int x) {
        if(c.length-1 == x) {
            list1.add(String.valueOf(c));
            return;
        }
        HashSet<Character> set = new HashSet<>();
        for(int i=x;i<c.length;i++) {
            if(set.contains(c[i])) continue;
            set.add(c[i]);
            swap(x,i);
            dfs(x+1);
            swap(x,i);
        }
    }

    private void swap(int a, int b) {
        char temp = c[a];
        c[a] = c[b];
        c[b] = temp;
    }

    //39 摩尔投票法
    public int majorityElement(int[] nums) {
        int sum = 0,x = 0;
        for(int n : nums) {
            if(sum == 0) x = n;
            sum += n == x ? 1 : -1;
        }
        return x;
    }

    //40
//    // 计数排序
//    public int[] getLeastNumbers(int[] arr, int k) {
//        int min = arr[0],max = arr[0];
//        for(int n : arr) {
//            if(n<min) min = n;
//            if(n>max) max = n;
//        }
//        int[] temp = new int[max-min+1];
//        for(int n : arr) {
//            temp[n-min]++;
//        }
//        int index = 0;
//        for(int i=0;i<max-min+1;i++) {
//            while(temp[i]-- != 0) {
//                arr[index++] = i+min;
//            }
//        }
//        int[] res = Arrays.copyOf(arr,k);
//        return res;
//    }
    //快排
    public int[] getLeastNumbers(int[] arr, int k) {
        if(k==0 || arr.length==0) return new int[0];
        quick(arr,0,arr.length-1,k);
        int[] res = Arrays.copyOf(arr,k);
        return res;
    }

    private void quick(int[] arr, int l, int r, int k) {
        int temp = arr[l];
        int i=l, j=r;
        while(i<j) {
            while(arr[j] >= temp && i<j) j--;
            if(arr[j]<temp) arr[i++] = arr[j];
            while(arr[i] <= temp && i<j) i++;
            if(arr[i]>temp) arr[j--] = arr[i];
        }
        arr[i] = temp;
        if(i-l+1 == k) return;
        else if(i-l+1 < k) quick(arr, i+1,r,k-(i-l+1));
        else quick(arr,l,i-1,k);
    }

    //42
    public int maxSubArray(int[] nums) {
        int sum = 0;
        int max = nums[0];
        for(int i=0;i<nums.length;i++) {
            sum += nums[i];
            if(sum<0) sum = 0;
            if(sum>max) max = sum;
        }
        return sum;
    }

    //43
    public int countDigitOne(int n) {
        int digit = 1,high = n/10,low = 0,cur = n%10;
        int res = 0;
        while(cur!=0 || high != 0) {
            if(cur == 0) {
                res += high*digit;
            }
            else if(cur == 1) {
                res += high*digit + low + 1;
            }
            else {
                res += (high + 1)*digit;
            }
            cur = high %10;
            digit *= 10;
            high /= 10;
            low = n%digit;
        }
        return res;
    }

    //44
    public int findNthDigit(int n) {
        int digit = 1;
        long start = 1;
        while(( n- digit*start*9)>0) {
            n -= digit*start*9;
            digit ++;
            start *= 10;
        }
        int cnt = n/digit;
        n -= cnt*digit;
        if(n>0) cnt++;
        long num = start+cnt-1;
        return (int)num/(int)Math.pow(10,(digit-n)%digit)%10;
    }

    //45
    public String minNumber(int[] nums) {
        String[] strs = new String[nums.length];
        for(int i=0;i<nums.length;i++) {
            strs[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(strs,(x, y)->(x + y).compareTo(y + x));
        StringBuilder res = new StringBuilder();      //这里用StringBuilder 比用 StringBuffer 快
        for(String s : strs) {
            res.append(s);
        }
        return res.toString();
    }

    //46
    public int translateNum(int num) {
        char[] chx = String.valueOf(num).toCharArray();
        return divide(chx);
    }

    private int divide(char[] chx) {
        if(chx.length == 1) return 1;
        int a = chx[0]-'0';
        int b = chx[1]-'0'+a*10;
        if(b>=10 && b<=25) return divide(Arrays.copyOfRange(chx,1,chx.length)) + divide(Arrays.copyOfRange(chx,2,chx.length));
        return divide(Arrays.copyOfRange(chx,1,chx.length));
    }

    //47
    public int maxValue(int[][] grid) {
        int[][] res = new int[grid.length][grid[0].length];
        for(int i=0;i<grid.length;i++) {
            for(int j=0;j<grid[i].length;j++) {
                if(i==0 && j==0) res[i][j] = grid[i][j];
                else if(j==0) res[i][j] = res[i-1][j] + grid[i][j];
                else if(i==0) res[i][j] = grid[i][j] + res[i][j-1];
                else res[i][j] = Math.max(res[i-1][j], res[i][j-1]) + grid[i][j];
            }
        }
        return res[grid.length-1][grid[0].length-1];
    }

    //48
    public int lengthOfLongestSubstring(String s) {
        HashMap<Character,Integer> map = new HashMap();
        int i=-1,res=0;
        for(int j=0;j<s.length();j++) {
            if(map.containsKey(s.charAt(j))) {
                i = Math.max(i,map.get(s.charAt(j)));
            }
            map.put(s.charAt(j),j);
            res = Math.max(res,j-i);
        }
        return res;
    }

    //49
    public int nthUglyNumber(int n) {
        int[] dp = new int[n];
        dp[0] = 1;
        int a=0,b=0,c=0;
        for(int i=1;i<n;i++) {
            int n2=dp[a]*2, n3=dp[b]*3, n5=dp[c]*5;
            dp[i] = Math.min(Math.min(n2,n3),n5);
            if(dp[i] == n2) a++;
            if(dp[i] == n3) b++;
            if(dp[i] == n5) c++;
        }
        return dp[n-1];
    }

    //50
    public char firstUniqChar(String s) {
        int[] count = new int[26];
        for(char c : s.toCharArray()) {
            count[c-'a']++;
        }
        for(char c : s.toCharArray()) {
            if(count[c-'a'] == 1) return c;
        }
        return ' ';
    }

    //51
    public int reversePairs(int[] nums) {
        int[] temp = new int[nums.length];
        if(nums.length < 2) {
            return 0;
        }
        return reversePairs(nums, 0, nums.length-1, temp);
    }

    private int reversePairs(int[] nums, int left, int right, int[] temp) {
        if(left == right) {
            return 0;
        }
        int mid = (left + right) / 2;
        int leftPairs = reversePairs(nums, left, mid, temp);
        int rightPairs = reversePairs(nums, mid + 1, right , temp);
        if(nums[mid] <= nums[mid+1]) return leftPairs + rightPairs;
        int crossPairs = mergeAndCount(nums, left, mid, right, temp);
        return leftPairs + rightPairs + crossPairs;
    }

    private int mergeAndCount(int[] nums, int left, int mid, int right, int[] temp) {
        int i = left,j = mid+1, k = left, count = 0;
        while(i<=mid && j<=right) {
            if(nums[i] <= nums[j]) {
                temp[k++] = nums[i++];
                count += j-(mid+1);
            }
            else {
                temp[k++] = nums[j++];
                //count += j-(mid+1);
            }
        }
        while(i<=mid) {
            temp[k++] = nums[i++];
            count += j-(mid+1);
        }
        while(j<=right) {
            temp[k++] = nums[j++];
        }
        for(k=left;k<=right;k++) {
            nums[k] = temp[k];
        }
        return count;
    }

    //52
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode p = headA,q=headB;
        while(p != q) {
            p = p!=null ? p.next : headB;
            q = q!=null ? q.next : headA;
        }
        return p;
    }

    //53-1
    //二分法
//    public int search(int[] nums, int target) {
//        int l = 0, r = nums.length-1;
//        while(l<=r) {
//            int mid = (l+r)/2;
//            if(nums[mid] >= target) {
//                r = mid-1;
//            }
//            else l = mid+1;
//        }
//        int leftBound = l;
//        l = 0;
//        r = nums.length-1;
//        while(l<=r) {
//            int mid = (l+r)/2;
//            if(nums[mid] > target)  r=mid-1;
//            else l=mid+1;
//        }
//        return r-leftBound+1;
//    }
    public int search(int[] nums, int target) {
        return binarySearch(nums,target+0.5)-binarySearch(nums,target-0.5);
    }
    private int binarySearch(int[] nums, double target) {
        int l = 0, r = nums.length-1;
        while(l<=r) {
            int mid = (l+r)/2;
            if(nums[mid]>target) r = mid-1;
            else l = mid+1;
        }
        return l;
    }

    //53
    public int missingNumber(int[] nums) {
        int i=0,j=nums.length-1;
        while(i<=j) {
            int m = (i+j)/2;
            if(nums[m] == m) i = m+1;
            else j=m-1;
        }
        return i;
    }

    //54
    int res1,k;
    public int kthLargest(TreeNode root, int k) {
        this.k = k;
        inorder(root);
        return res1;
    }

    private void inorder(TreeNode root) {
        if(root==null || k==0) return;
        inorder(root.right);
        if(--k == 0) res1 = root.val;
        inorder(root.left);
    }

    //55-1
    public int maxDepth(TreeNode root) {
        if(root == null) return 0;
        if(root.left == null && root.right == null) return 1;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    //55-2
    public boolean isBalanced(TreeNode root) {
        if(root == null) return true;
        if(!isBalanced(root.left) || !isBalanced(root.right)) return false;
        if(Math.abs(maxDepth(root.left) - maxDepth(root.right)) > 1) return false;
        return true;
    }

    //56-1  时间复杂度O(n),空间复杂度O(1)
    public int[] singleNumbers(int[] nums) {
        int x = 0;
        for(int n : nums) {
            x ^= n;
        }
        int a=0, b=0, val=1;
        while((val&x) == 0) {
            val <<= 1;
        }
        for(int n: nums) {
            if((val&n) == 0) {
                a ^= n;
            }
            else {
                b ^= n;
            }
        }
        return new int[] {a, b};
    }

    //56-2
    public int singleNumber(int[] nums) {
        int twos=0, ones=0;
        for(int n : nums) {
            ones = ones^n&~twos;
            twos = twos^n&~ones;
        }
        return ones;
    }

    //57-1   对撞双指针法
    public int[] twoSum(int[] nums, int target) {
        int i=0, j=nums.length-1;
        while(i<j) {
            if(nums[i]+nums[j] < target) i++;
            else if(nums[i]+nums[j] > target) j--;
            else return new int[]{nums[i], nums[j]};
        }
        return new int[0];
    }

    //57-2
    public int[][] findContinuousSequence(int target) {
        LinkedList<int[]> list = new LinkedList<>();
        int i = 1;
        while(target > 0) {
            target -= i++;
            if(target>0 && target%i==0) {
                int[] nums = new int[i];
                for(int k=target/i,j=0;k<target/i+i;k++,j++) {
                    nums[j] = k;
                }
                list.add(nums);
            }
        }
        Collections.reverse(list);
        return list.toArray(new int[0][]);
    }

    //58-1
    public String reverseWords(String s) {
        String[] strs = s.split(" ");
        StringBuilder res = new StringBuilder();
        for(int i=strs.length-1;i>=0;i--) {
            if(strs[i].equals("")) continue;
            res.append(strs[i] + " ");
        }
        return res.toString().trim();    //去除首尾空格
    }

    //58-2
    public String reverseLeftWords(String s, int n) {
        StringBuilder res = new StringBuilder();
        res.append(s.substring(n,s.length()-n));
        res.append(s.substring(0,n));
        return res.toString();
    }

    //59-1   双端队列
    //方法一  1ms
//    public int[] maxSlidingWindow(int[] nums, int k) {
//        int len = nums.length;
//        if (len == 0){
//            return new int[0];
//        }
//        int[] res = new int[len-k+1];
//        int index = -1,max = Integer.MIN_VALUE;
//        for(int i=0;i<len-k+1;i++) {
//            if(index >= i) {
//                if(max<nums[i+k-1]) {
//                    max = nums[i+k-1];
//                    index = i+k-1;
//                }
//            }
//            else {
//                max = Integer.MIN_VALUE;
//                for(int j=i;j<i+k;j++) {
//                    if(max < nums[j]) {
//                        max = nums[j];
//                        index = j;
//                    }
//                }
//            }
//            res[i] = max;
//        }
//        return res;
//    }

    //方法二  14ms
    public int[] maxSlidingWindow(int[] nums, int k) {
        if(nums.length == 0 || k == 0) return new int[0];
        Deque<Integer> deque = new LinkedList<>();
        int[] res = new int[nums.length-k+1];
        for(int i=0;i<k;i++) {
            while(!deque.isEmpty() && deque.peekLast() < nums[i]) {
                deque.removeLast();
            }
            deque.addLast(nums[i]);
        }
        res[0] = deque.peekFirst();
        for(int i=k;i<nums.length;i++) {
            if(deque.peekFirst() == nums[i-k]) deque.removeFirst();
            while(!deque.isEmpty() && deque.peekLast() < nums[i]) {
                deque.removeLast();
            }
            deque.addLast(nums[i]);
            res[i-k+1] = deque.peekFirst();
        }
        return res;
    }

    //60
    public double[] twoSum(int n) {
        double[] res = {1/6d,1/6d,1/6d,1/6d,1/6d,1/6d};
        for(int i=2;i<=n;i++) {
            double[] temp = new double[5*i+1];
            for(int j=0;j<res.length;j++) {
                for(int x=0;x<6;x++) {
                    temp[j+x] += res[j]/6;
                }
            }
            res = temp;
        }
        return res;
    }

    //61
    public boolean isStraight(int[] nums) {
        Set<Integer> repeat = new HashSet<>();
        int max=0, min = 14;
        for(int i=0;i<5;i++) {
            if(nums[i]==0) continue;
            if(repeat.contains(nums[i])) return false;
            repeat.add(nums[i]);
            max = Math.max(max, nums[i]);
            min = Math.min(min, nums[i]);
        }
        return max - min < 5;
    }

    //62
    //递归
//    public int lastRemaining(int n, int m) {
//        return f(n, m);
//    }
//
//    private int f(int n, int m) {
//        if(n==1) return 0;
//        int x = f(n-1, m);
//        return (m+x)%n;
//    }

    //递归改为迭代
    public int lastRemaining(int n, int m) {
        int f=0;
        for(int i=2;i<=n;i++) {
            f = (m+f)%i;
        }
        return f;
    }

    //63
    public int maxProfit(int[] prices) {
        int res = 0,min = Integer.MAX_VALUE;
        for(int price : prices) {
            min = Math.min(min, price);
            res = Math.max(res, price-min);
        }
        return res;
    }

    //64
    public int sumNums(int n) {
        boolean flag = n > 1 && (n+=sumNums(n-1)) >0;
        return n;
    }

    //65
    public int add(int a, int b) {
        while(b!=0) {
            int c = (a&b) << 1;
            a = a^b;
            b = c;
        }
        return a;
    }

    //66
    public int[] constructArr(int[] a) {
        if(a.length==0) return new int[0];
        int[] b = new int[a.length];
        int temp=1;
        b[0]=1;
        for(int i=1;i<a.length;i++) {
            b[i] = b[i-1] * a[i-1];
        }
        for(int i=a.length-2;i>=0;i--) {
            temp *= a[i+1];
            b[i] *= temp;
        }
        return b;
    }

    //67
    public int strToInt(String str) {
        char[] chs = str.trim().toCharArray();
        if(chs.length==0) return 0;
        long res=0;
        int flag=1,i=0;
        if(chs[0]=='-') {
            flag=-1;
            i++;
        }
        else if(chs[0]=='+') i++;
        for(;i<chs.length;i++) {
            if(chs[i]>='0' && chs[i]<='9') {
                res = chs[i]-'0' + res*10;
                if(flag==1 && res > Integer.MAX_VALUE) {
                    return Integer.MAX_VALUE;
                }
                else if(flag==-1 && res*-1 < Integer.MIN_VALUE) {
                    return Integer.MIN_VALUE;
                }
            }
            else break;
        }
        if(flag==0) flag=1;
        return (int)res*flag;
    }

    //68-1
    public TreeNode lowestCommonAncestor1(TreeNode root, TreeNode p, TreeNode q) {
        if(p.val > q.val) {
            TreeNode t = p;
            p = q;
            q = t;
        }
        while(root != null) {
            if(q.val<root.val) root = root.left;
            else if(p.val > root.val) root = root.right;
            else break;
        }
        return root;
    }

    //68-2
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root==null || root==p || root==q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if(left==null) return right;
        if(right==null) return left;
        return root;
    }

}
