package com.muqi.leetcode.test2;

/**
 * @author muqi
 * @since 2020/7/9 17:44
 */
public class Trie {
    public Trie[] next;
    public boolean isEnd;

    public Trie() {
        next = new Trie[26];
        isEnd = false;
    }

    public void insert(String word) {
        Trie curPos = this;
        for(int i=word.length()-1;i>=0;i--) {
            int t = word.charAt(i) - 'a';
            if(curPos.next[t]==null) {
                curPos.next[t] = new Trie();
            }
            curPos = curPos.next[t];
        }
        curPos.isEnd = true;
    }
}
