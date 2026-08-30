"""
Servidor local do dashboard.

Postura de seguranca (intencional, nao mexer sem querer o efeito):
  * bind fixo em 127.0.0.1 -> so processos DESTA maquina alcancam. Nao e
    acessivel pela rede local nem pela internet, independente do firewall.
  * serve apenas arquivos estaticos de web/. Nao ha rota POST, webhook,
    callback ou qualquer caminho que aceite comando de terceiro.
  * nenhuma chamada de rede de saida: o navegador le somente data.json local.

Uso:
    python serve.py            # porta 8765, abre o navegador
    python serve.py --porta 9000 --sem-navegador
"""

import argparse
import http.server
import sys
import webbrowser
from functools import partial
from pathlib import Path

HOST = "127.0.0.1"  # nunca 0.0.0.0
WEB = Path(__file__).resolve().parent / "web"


class Handler(http.server.SimpleHTTPRequestHandler):
    """Estatico, sem cache, sem log ruidoso."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        # o dashboard e 100% local; proibir qualquer recurso externo
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'",
        )
        super().end_headers()

    def log_request(self, code="-", size="-"):
        if not str(code).startswith("2"):  # so ruido de verdade (4xx/5xx)
            sys.stderr.write(f"  {code}  {self.requestline}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--porta", type=int, default=8765)
    ap.add_argument("--sem-navegador", action="store_true")
    args = ap.parse_args()

    if not (WEB / "data.json").exists():
        print("data.json nao encontrado. Rode primeiro:  python fetch_data.py",
              file=sys.stderr)
        return 1

    try:
        srv = http.server.ThreadingHTTPServer(
            (HOST, args.porta), partial(Handler, directory=str(WEB))
        )
    except OSError as e:
        if e.errno in (48, 98, 10048):
            print(f"Porta {args.porta} ja em uso. Tente:  python serve.py --porta {args.porta + 1}",
                  file=sys.stderr)
            return 1
        raise

    url = f"http://{HOST}:{args.porta}/"
    print(f"\n  Dashboard em {url}")
    print(f"  Acessivel somente desta maquina ({HOST}).")
    print("  Ctrl+C para parar.\n")

    if not args.sem_navegador:
        webbrowser.open(url)

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  Encerrado.")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
