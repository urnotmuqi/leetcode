package com.muqi.leetcode.test;

import java.util.LinkedList;
import java.util.Queue;

/**
 * @author muqi
 * @since 2020/5/19 17:23
 *
 * 37
 */
public class Codec {
    public String serialize(TreeNode root) {
        if(root == null) return "[]";
        Queue<TreeNode> q = new LinkedList<>();
        StringBuffer res = new StringBuffer("[");
        q.add(root);
        while(!q.isEmpty()) {
            TreeNode p = q.poll();
            if(p != null) {
                res.append(String.valueOf(p.val) + ',');
                q.add(p.left);
                q.add(p.right);
            }
            else res.append("null,");
        }
        res.deleteCharAt(res.length()-1);
        res.append("]");
        return res.toString();
    }

    public TreeNode deserialize(String data) {
        if(data == null || data.equals("[]")) return null;
        String[] s = data.substring(1,data.length()-1).split(",");
        Queue<TreeNode> q = new LinkedList<>();
        TreeNode head = new TreeNode(Integer.valueOf(s[0]));
        q.add(head);
        for(int i=1;i<s.length;i=i+2) {
            TreeNode p = q.poll();
            if(!s[i].equals("null")) {
                p.left = new TreeNode(Integer.valueOf(s[i]));
                q.add(p.left);
            }
            if(!s[i+1].equals("null")) {
                p.right = new TreeNode(Integer.valueOf(s[i+1]));
                q.add(p.right);
            }
        }
        return head;
    }
}
