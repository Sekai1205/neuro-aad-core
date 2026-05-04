import socket
import time

def main():
    print("=== Python Slow Loop (MAML Brain) ===")
    
    # 1. UDPソケット（トンネルの入り口）を作成
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 送信先は自分のMac（localhost）の5005番ポート
    server_address = ('127.0.0.1', 5005)
    
    print("🧠 C++エンジンに向けて、AIの予測結果を送信します。")
    print("Ctrl+C で終了します。\n")

    try:
        # AIが2秒に1回、脳波を解析して指令を出すシミュレーション
        for i in range(1, 10):
            # 今回はテストなので、LとRを交互に送信してみます
            command = "L" if i % 2 == 0 else "R"
            
            print(f"[{i}] AI予測: {command} に注意が向いています -> C++へ送信！")
            
            # 文字列をバイトデータ（b"L"など）に変換して送信
            sock.sendto(command.encode(), server_address)
            
            # Slow Loopの周期（2秒間隔）
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n送信を停止しました。")
    finally:
        sock.close()

if __name__ == '__main__':
    main()