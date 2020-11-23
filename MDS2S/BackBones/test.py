# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        copy = ListNode(0)
        p = copy
        while True:
            count = k
            stack = []
            tmp = head
            while count and tmp:
                stack.append(tmp)
                tmp = tmp.next
                count -= 1
            if count:
                p.next = head
                break
            while stack:
                p.next = stack.pop()
                p = p.next
            p.next = tmp
            head = tmp

        return copy.next

    def test(self, head, k):
        head = self.reverseKGroup(head, k)
        while head is not None:
            print(head.val)
            head = head.next


head = ListNode(5, None)
head = ListNode(4, head)
head = ListNode(3, head)
head = ListNode(2, head)
head = ListNode(1, head)
s = Solution()
s.test(head, 3)