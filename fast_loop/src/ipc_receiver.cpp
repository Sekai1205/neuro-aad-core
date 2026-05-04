#include <iostream>
#include <string>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <thread>
#include <chrono>
#include <iomanip>

#define PORT 5005

// バーチャル音量メーターを描画する関数
void print_volume_meter(float gain_L, float gain_R) {
    int bars_L = static_cast<int>(gain_L * 20);
    int bars_R = static_cast<int>(gain_R * 20);
    std::cout << "\r[L] " << std::string(bars_L, '#') << std::string(20 - bars_L, ' ') << " (" << std::fixed << std::setprecision(2) << gain_L << ") | "
              << "[R] " << std::string(bars_R, '#') << std::string(20 - bars_R, ' ') << " (" << std::fixed << std::setprecision(2) << gain_R << ")" << std::flush;
}

int main() {
    std::cout << "=== C++ Fast Loop (Smooth Audio Engine) ===" << std::endl;
    
    int sockfd;
    struct sockaddr_in servaddr, cliaddr;
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) return -1;
    fcntl(sockfd, F_SETFL, O_NONBLOCK); // 止まらない耳（非同期）
    
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = INADDR_ANY;
    servaddr.sin_port = htons(PORT);
    if (bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) return -1;

    // --- 音量管理の変数 ---
    // 最初は左右ともフラット（0.5）からスタート
    float current_gain_L = 0.5f, target_gain_L = 0.5f;
    float current_gain_R = 0.5f, target_gain_R = 0.5f;
    
    // スムージング係数（小さいほどゆっくり変化する。今回は視覚的に分かりやすいスピードに設定）
    float alpha = 0.1f; 

    std::cout << "🎧 UDPポート " << PORT << " で待機中... 音量スムージング[ON]" << std::endl;

    while (true) {
        char buffer[16];
        socklen_t len = sizeof(cliaddr);
        int n = recvfrom(sockfd, (char *)buffer, sizeof(buffer)-1, MSG_DONTWAIT, (struct sockaddr *) &cliaddr, &len);
        
        // 脳（Python）から指令が来たら、「目標値（Target）」だけを書き換える
        if (n > 0) {
            buffer[n] = '\0';
            std::cout << "\n⚡️ 指令受信: [" << buffer << "] -> 目標音量を変更します" << std::endl;
            if (buffer[0] == 'L') {
                target_gain_L = 1.0f; // 左を最大に
                target_gain_R = 0.1f; // 右を最小に
            } else if (buffer[0] == 'R') {
                target_gain_L = 0.1f;
                target_gain_R = 1.0f;
            }
        }

        // --- 超高速オーディオループ（ここが実際のサンプルごとの処理） ---
        // 目標値に向かって、現在値を少しずつ滑らかに近づける（ポップノイズ防止）
        current_gain_L += (target_gain_L - current_gain_L) * alpha;
        current_gain_R += (target_gain_R - current_gain_R) * alpha;

        // 状態を画面に出力
        print_volume_meter(current_gain_L, current_gain_R);

        // 実際の音声処理のバッファ周期（約10ms）を模擬
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    close(sockfd);
    return 0;
}