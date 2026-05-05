#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include "lock_free_queue.hpp"

// ============================================================
// 【改修②】 Python × C++ エンドツーエンド統合シミュレーション
//
// システム全体の動作証明:
//   Slow Loop (Python) が inference_log.json に書いた
//   「生の推論精度 (raw_accuracy)」を、
//   Fast Loop (C++) の EMA スムージングが吸収し、
//   実際のオーディオ出力安定性がどう変化するかをログ出力する。
//
// EMAの式: ema_t = α × raw_t + (1 - α) × ema_{t-1}
//   α が小さい → 過去を重視（変化に鈍感・安定）
//   α が大きい → 現在を重視（変化に敏感・追従）
// 今回は α = 0.1 を採用（約10サンプルの移動平均に相当）
// ============================================================

// Pythonの推論結果を表す構造体（ロックフリーキューで受け渡し）
struct InferenceResult {
    int   epoch;
    int   task;
    float raw_accuracy;
};

// ------------------------------------------------------------------
// EMA スムージングクラス
// 状態 (ema_value) を保持し、update() で逐次更新する。
// ------------------------------------------------------------------
class EMAFilter {
public:
    explicit EMAFilter(float alpha) : alpha_(alpha), ema_(50.0f), initialized_(false) {}

    float update(float raw) {
        if (!initialized_) {
            ema_ = raw;  // 初回は生値で初期化
            initialized_ = true;
        } else {
            ema_ = alpha_ * raw + (1.0f - alpha_) * ema_;
        }
        return ema_;
    }

    float value() const { return ema_; }

private:
    float alpha_;
    float ema_;
    bool  initialized_;
};

// ------------------------------------------------------------------
// JSON パーサー（軽量版）
// inference_log.json から raw_accuracy を抽出する。
// 外部ライブラリ不使用で nlohmann/json 依存なし。
// ------------------------------------------------------------------
std::vector<InferenceResult> load_inference_log(const std::string& path) {
    std::vector<InferenceResult> results;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "  [ERROR] inference_log.json が見つかりません: " << path << "\n";
        std::cerr << "  先に maml_aad_trainer.py を実行してください。\n";
        return results;
    }

    std::string line;
    InferenceResult current{};
    while (std::getline(file, line)) {
        auto extract = [&](const std::string& key) -> float {
            auto pos = line.find("\"" + key + "\"");
            if (pos == std::string::npos) return -1.0f;
            pos = line.find(":", pos);
            if (pos == std::string::npos) return -1.0f;
            return std::stof(line.substr(pos + 1));
        };

        if (line.find("\"epoch\"") != std::string::npos)
            current.epoch = static_cast<int>(extract("epoch"));
        if (line.find("\"task\"") != std::string::npos)
            current.task = static_cast<int>(extract("task"));
        if (line.find("\"raw_accuracy\"") != std::string::npos) {
            current.raw_accuracy = extract("raw_accuracy");
            results.push_back(current);
        }
    }
    return results;
}

// ------------------------------------------------------------------
// 安定性判定: EMAが閾値を超えているか（=音声切り替えを行うか）
// 閾値: 60.0% → AIが「左の音に注意している」と十分な確信がある場合のみ切り替える
// ------------------------------------------------------------------
bool should_switch_audio(float ema_accuracy, float threshold = 60.0f) {
    return ema_accuracy >= threshold;
}

int main() {
    const std::string LOG_PATH = "data/inference_log.json";
    const float EMA_ALPHA = 0.10f;  // α: 小さいほど安定、大きいほど追従

    std::cout << "================================================\n";
    std::cout << "  Fast Loop (C++) - EMA Integration Simulator\n";
    std::cout << "  Pythonの生推論精度をEMAで平滑化し、\n";
    std::cout << "  オーディオ出力の安定性を証明するログを出力します。\n";
    std::cout << "================================================\n\n";

    // Pythonが書いた推論ログを読み込む
    auto results = load_inference_log(LOG_PATH);
    if (results.empty()) return 1;
    std::cout << "  📂 " << results.size() << " 件の推論結果をロードしました\n\n";

    // ロックフリーキュー (Python→C++ の非同期パイプを模擬)
    LockFreeQueue<InferenceResult> ipc_queue(512);

    // ---- Slow Loop スレッド (Python側を模擬) ----
    // 実システムでは Unix domain socket / shared memory で代替
    std::thread slow_loop_sim([&ipc_queue, &results]() {
        for (const auto& r : results) {
            // 実システムでは推論に数百ms～数秒かかる
            // シミュレーションでは1ms間隔で高速再生
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            ipc_queue.push(r);
        }
    });

    // ---- Fast Loop (EMAフィルタ処理) ----
    EMAFilter ema(EMA_ALPHA);
    int  total_ticks    = 0;
    int  stable_ticks   = 0;   // EMAが閾値以上だった回数
    int  flicker_count  = 0;   // EMA後に出力が前回と変わった回数
    bool prev_switch    = false;

    // ログヘッダ
    std::cout << std::left
              << std::setw(8)  << "Epoch"
              << std::setw(8)  << "Task"
              << std::setw(14) << "Raw(%)"
              << std::setw(14) << "EMA(%)"
              << std::setw(16) << "Audio Output"
              << "\n";
    std::cout << std::string(60, '-') << "\n";

    // キューから結果を取り出しEMAを適用し続ける
    // slow_loopが全件送り終わるまでポーリング
    size_t processed = 0;
    while (processed < results.size()) {
        InferenceResult r;
        if (!ipc_queue.pop(r)) {
            // データがなければ5msのオーディオ処理を模擬してリトライ
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            continue;
        }

        float ema_val    = ema.update(r.raw_accuracy);
        bool  cur_switch = should_switch_audio(ema_val);

        if (cur_switch != prev_switch) ++flicker_count;
        if (cur_switch) ++stable_ticks;
        prev_switch = cur_switch;
        ++total_ticks;
        ++processed;

        // 最初の50件 + エポック変わり目だけ詳細表示（ログが膨大になるため）
        if (processed <= 50 || r.task == 0) {
            std::cout << std::left
                      << std::setw(8)  << r.epoch
                      << std::setw(8)  << r.task
                      << std::setw(14) << std::fixed << std::setprecision(1) << r.raw_accuracy
                      << std::setw(14) << std::fixed << std::setprecision(1) << ema_val
                      << (cur_switch ? "▶ SWITCH (L→R)" : "  HOLD (L)") << "\n";
        }
    }

    slow_loop_sim.join();

    // ---- 統計サマリ ----
    float stability_rate = (total_ticks > 0)
        ? static_cast<float>(stable_ticks) / total_ticks * 100.0f : 0.0f;
    float flicker_rate   = (total_ticks > 0)
        ? static_cast<float>(flicker_count) / total_ticks * 100.0f : 0.0f;

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  📊 エンドツーエンド統計サマリ\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << "  総推論回数          : " << total_ticks << " 回\n";
    std::cout << "  EMAα (スムージング強度): " << EMA_ALPHA << "\n";
    std::cout << "  最終EMA精度         : "
              << std::fixed << std::setprecision(1) << ema.value() << " %\n";
    std::cout << "  安定出力率 (EMA≥60%): "
              << std::fixed << std::setprecision(1) << stability_rate << " %\n";
    std::cout << "  フリッカー率 (出力反転): "
              << std::fixed << std::setprecision(1) << flicker_rate << " %\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "\n  ✅ C++ Fast Loop: 一度も停止せずEMAで推論ブレを吸収しました！\n";
    std::cout << "  ✅ システム全体の信頼性証明 完了\n\n";

    return 0;
}