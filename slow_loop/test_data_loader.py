import scipy.io as sio
import os

def main():
    print("=== KULeuven EEG Dataset Loader Test ===")
    
    # S1.mat (被験者1のデータ) へのパスを指定
    # 実行場所が neuro-aad-core フォルダであることを想定
    file_path = "data/KULeuven data set/S1.mat"
    
    if not os.path.exists(file_path):
        print(f"❌ エラー: {file_path} が見つかりません。")
        print("'data' フォルダの中に 'S1.mat' が正しく入っているか確認してください。")
        return

    print("S1.mat を読み込んでいます... (ファイルが大きいので数秒かかります)\n")
    
    try:
        # .matファイルを読み込む
        mat_contents = sio.loadmat(file_path)
        print("✅ 読み込み大成功！ファイルの中に以下のデータ（鍵）が入っています：")
        
        # 中身の構造（Key）を表示
        for key in mat_contents.keys():
            # Pythonのシステム的な隠しデータ（__から始まるもの）以外を表示
            if not key.startswith('__'):
                data_type = type(mat_contents[key]).__name__
                # もし配列ならサイズ（shape）も表示
                if hasattr(mat_contents[key], 'shape'):
                    print(f" 🧠 - {key}: 型={data_type}, サイズ={mat_contents[key].shape}")
                else:
                    print(f" 🧠 - {key}: 型={data_type}")
                    
        print("\nこれでAIモデルに本物の脳波を流し込む準備が整いました！")

    except Exception as e:
        print(f"❌ 読み込み中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()