package com.muqi.leetcode.test2;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

/**
 * @author muqi
 * @since 2020/6/30 9:25
 */
public class Solution {
    //1.1
    public boolean isUnique(String astr) {
        HashSet<Character> set = new HashSet<>();
        for(int i=0;i<astr.length();i++) {
            set.add(astr.charAt(i));
        }
        if(set.size() != astr.length()) return false;
        return true;
    }

    //1.2
    public boolean CheckPermutation(String s1, String s2) {
        int len1 = s1.length(), len2 = s2.length();
        if(len1 != len2) return false;
        char[] chs1 = s1.toCharArray();
        char[] chs2 = s2.toCharArray();
        Arrays.sort(chs1);
        Arrays.sort(chs2);
        for(int i=0;i<len1;i++) {
            if(chs1[i] != chs2[i]) return false;
        }
        return true;
    }

    //1.3
    public String replaceSpaces(String S, int length) {
        char[] chs = S.toCharArray();
        int i=length-1, j=S.length()-1;
        while(i>=0) {
            if(chs[i] == ' ') {
                chs[j--] = '0';
                chs[j--] = '2';
                chs[j--] = '%';
            }
            else chs[j--] = chs[i];
            i--;
        }
        return String.valueOf(chs, j+1, S.length()-j-1);
    }

    //17.13
    public int respace(String[] dictionary, String sentence) {
        int n = sentence.length();
        Trie root = new Trie();
        for(int i=0;i<dictionary.length;i++) {
            root.insert(dictionary[i]);
        }
        int[] dp = new int[n+1];
        dp[0] = 0;
        for(int i=1;i<=n;i++) {
            dp[i] = dp[i-1]+1;
            Trie curPos = root;
            for(int j=i;j>0;j--) {
                int t = sentence.charAt(j-1) - 'a';
                if(curPos.next[t]==null) {
                    break;
                }else if(curPos.next[t].isEnd) {
                    dp[i] = dp[j-1];
                }
                if(dp[i]==0) break;
                curPos = curPos.next[t];
            }
        }
        return dp[n];
    }
}
