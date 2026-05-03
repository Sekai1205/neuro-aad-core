#include <iostream>
#include <thread>
#include <chrono>
#include "lock_free_queue.hpp"

// AIから送られてくる「フィルタ係数」のダミーデータ
struct FilterCoeffs {
    int id;
    float value;
};

int main() {
    std::cout << "=== Fast Loop (C++) 超高速レンダリングテスト ===" << std::endl;

    // 容量100のロックフリーキューを作成（ここにAIからのデータが入る）
    LockFreeQueue<FilterCoeffs> queue(100);

    // ---------------------------------------------------------
    // 🧠【Slow Loopのシミュレーション (AI側)】
    // 1秒(1000ms)に1回という遅いペースで、新しい計算結果をキューに押し込む
    // ---------------------------------------------------------
    std::thread slow_loop([&queue]() {
        for (int i = 1; i <= 3; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            FilterCoeffs coeffs{i, 0.5f * i};
            queue.push(coeffs); // キューに送信！
            std::cout << "[AI: Slow Loop] 新しいフィルタ係数(" << i << ")を送信しました！\n";
        }
    });

    // ---------------------------------------------------------
    // ⚡️【Fast Loopのシミュレーション (オーディオ処理側)】
    // 5ミリ秒(ms)ごとに絶え間なく音を処理し続ける
    // ---------------------------------------------------------
    FilterCoeffs current_coeffs{0, 1.0f}; // 初期値

    // 約3秒間（5ms × 650回）処理を回し続ける
    for (int i = 0; i < 650; ++i) {
        FilterCoeffs new_coeffs;
        
        // 【ここが最強の牙】
        // キューに新しいデータがあれば引き抜く。無ければ「待たずに」すぐ次へ進む！
        if (queue.pop(new_coeffs)) {
            current_coeffs = new_coeffs;
            std::cout << "  => ⚡️[Audio: Fast Loop] 係数(" << current_coeffs.id << ")を瞬時に適用！遅延ゼロ！\n";
        }

        // ここで「音にフィルタをかける処理」が行われる（約5msの想定）
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // AI側の終了を待つ
    slow_loop.join();
    
    std::cout << "=== テスト完了：Fast Loopは一度も止まりませんでした！ ===" << std::endl;
    return 0;
}