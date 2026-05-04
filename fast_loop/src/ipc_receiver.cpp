#include <iostream>
#include <string>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <thread>
#include <chrono>

#define PORT 5005 // 通信に使う秘密のポート番号

int main() {
    std::cout << "=== C++ Fast Loop (Audio Engine) ===" << std::endl;
    std::cout << "🎧 UDPポート " << PORT << " で脳(Python)からの指令を待機中..." << std::endl;

    // 1. UDPソケット（トンネルの出口）を作成
    int sockfd;
    struct sockaddr_in servaddr, cliaddr;
    
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("Socket creation failed");
        return -1;
    }

    // 2. ノンブロッキング（非同期）モードに設定！ここが超重要！
    // これをしないと、Pythonから指令が来るまでC++の音声処理がフリーズしてしまいます
    fcntl(sockfd, F_SETFL, O_NONBLOCK);

    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = INADDR_ANY;
    servaddr.sin_port = htons(PORT);

    if (bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
        perror("Bind failed");
        return -1;
    }

    // 3. 超高速オーディオループ（擬似）
    while (true) {
        // --- 実際のシステムではここで音のフィルタリング処理が入ります ---
        
        // 脳(Python)から新しい指令(L or R)が来ていないかチラッと確認
        char buffer[16];
        socklen_t len = sizeof(cliaddr);
        int n = recvfrom(sockfd, (char *)buffer, sizeof(buffer)-1, MSG_DONTWAIT, (struct sockaddr *) &cliaddr, &len);
        
        if (n > 0) {
            buffer[n] = '\0'; // 文字列の終端をセット
            std::cout << "⚡️ 脳からの指令を受信: [" << buffer << "] -> オーディオフィルタを切り替えます！" << std::endl;
        }

        // CPUが爆発しないように10ミリ秒（0.01秒）だけ待つ（Fast Loopの周期）
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    close(sockfd);
    return 0;
}