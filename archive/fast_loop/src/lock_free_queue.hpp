#pragma once
#include <atomic>
#include <vector>
#include <cstddef>

// オーディオスレッドを絶対にブロックしないためのSPSCロックフリーキュー
template <typename T>
class LockFreeQueue {
private:
    std::vector<T> buffer_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
    size_t capacity_;

public:
    explicit LockFreeQueue(size_t capacity) 
        : buffer_(capacity), head_(0), tail_(0), capacity_(capacity) {}

    bool push(const T& item) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % capacity_;
        if (next_tail == head_.load(std::memory_order_acquire)) return false;
        buffer_[current_tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        if (current_head == tail_.load(std::memory_order_acquire)) return false;
        item = buffer_[current_head];
        head_.store((current_head + 1) % capacity_, std::memory_order_release);
        return true;
    }
};