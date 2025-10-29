import os
import sys
import subprocess
import venv
import platform
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, ".venv")
REQUIREMENTS_FILE = os.path.join(BASE_DIR, "requirements.txt")
MAIN_SCRIPT = os.path.join(BASE_DIR, "src", "main.py")

FAISS_INDEX_DIR = os.path.join(BASE_DIR, "storage")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")


def create_virtualenv():
    if not os.path.exists(VENV_DIR):
        print("📦 Criando ambiente virtual (.venv)...")
        venv.create(VENV_DIR, with_pip=True)
        print("✅ Ambiente virtual criado.")
    else:
        print("✅ Ambiente virtual já existe.")


def get_venv_paths():
    if platform.system() == "Windows":
        python_exec = os.path.join(VENV_DIR, "Scripts", "python.exe")
        pip_exec = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else:
        python_exec = os.path.join(VENV_DIR, "bin", "python")
        pip_exec = os.path.join(VENV_DIR, "bin", "pip")
    return python_exec, pip_exec


def run_in_venv(command):
    python_exec, pip_exec = get_venv_paths()
    if command[0] == "python":
        command[0] = python_exec
    elif command[0] == "pip":
        command[0] = pip_exec
    subprocess.check_call(command)


def install_requirements():
    print("📥 Atualizando pip...")
    run_in_venv(["python", "-m", "pip", "install", "--upgrade", "pip"])

    print("📦 Instalando dependências...")
    run_in_venv(["pip", "install", "-r", REQUIREMENTS_FILE])

    print("✅ Dependências instaladas.")


def check_ollama():
    python_exec, _ = get_venv_paths()
    code = """
import requests
try:
    r = requests.get('http://localhost:11434/api/tags', timeout=5)
    exit(0 if r.status_code == 200 else 1)
except Exception:
    exit(1)
"""
    result = subprocess.run([python_exec, "-c", code],
                            capture_output=True, text=True, timeout=10)
    return result.returncode == 0


def start_ollama():
    print("🚀 Tentando iniciar Ollama...")
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["ollama", "serve"],
                             creationflags=subprocess.CREATE_NEW_CONSOLE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["ollama", "serve"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        import time
        for i in range(10):
            time.sleep(1)
            if check_ollama():
                return True
            print(f"   Tentativa {i+1}/10...")
        print("⚠️  Ollama não respondeu após 10s.")
        return False
    except FileNotFoundError:
        print("❌ Ollama não encontrado. Instale: https://ollama.ai/download")
        return False


def check_cache_status():
    print("\n📊 Status do Cache e Índice")
    print("=" * 60)

    if os.path.exists(FAISS_INDEX_DIR):
        index_files = [f for f in os.listdir(FAISS_INDEX_DIR)
                       if os.path.isfile(os.path.join(FAISS_INDEX_DIR, f))]
        size = sum(os.path.getsize(os.path.join(FAISS_INDEX_DIR, f))
                   for f in index_files)
        print(f"✅ FAISS: {len(index_files)} arquivo(s), {size / (1024*1024):.2f} MB")
    else:
        print("⚠️  Nenhum índice FAISS encontrado.")

    if os.path.exists(CACHE_DIR):
        cache_size = sum(os.path.getsize(os.path.join(d, f))
                         for d, _, files in os.walk(CACHE_DIR) for f in files)
        print(f"✅ Cache embeddings: {cache_size / (1024*1024):.2f} MB")
    else:
        print("⚠️  Cache de embeddings não encontrado.")

    print("=" * 60 + "\n")


def rebuild_faiss_index():
    deleted = False
    if os.path.exists(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR)
        deleted = True
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        deleted = True
    print("✅ Índice e cache limpos." if deleted else "ℹ️ Nenhum cache encontrado.")


def check_python_version():
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    print(f"🐍 Python {version_str} detectado")
    if v < (3, 9):
        print("❌ Python 3.9+ necessário.")
        sys.exit(1)
    elif v >= (3, 12):
        print("⚠️  Python 3.12+ pode ter incompatibilidades. Recomendado 3.11.x")


def run_main():
    if not os.path.exists(MAIN_SCRIPT):
        print(f"❌ Arquivo não encontrado: {MAIN_SCRIPT}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🚀 Iniciando RAG Chatbot...")
    print("=" * 60 + "\n")
    run_in_venv(["python", MAIN_SCRIPT])


def main():
    rebuild_flag = "--rebuild" in sys.argv
    status_flag = "--status" in sys.argv

    print("=" * 60)
    print("🚀 Inicializando ambiente RAG Chatbot")
    print("=" * 60 + "\n")

    if status_flag:
        check_cache_status()
        sys.exit(0)

    check_python_version()
    create_virtualenv()
    install_requirements()

    if rebuild_flag:
        rebuild_faiss_index()
    else:
        check_cache_status()

    print("\n🔍 Verificando Ollama...")
    if not check_ollama():
        print("⚠️  Ollama não está ativo.")
        if not start_ollama():
            print("\n❌ Não foi possível iniciar o Ollama automaticamente.")
            print("   Inicie manualmente com: ollama serve")
            sys.exit(1)

    print("✅ Ollama está rodando.")
    run_main()


if __name__ == "__main__":
    main()
