"""
Arma un bundle de certificados que incluye los del almacén de certificados de Windows.

    python tools/ca_bundle_windows.py

Escribe `.certs/ca_bundle.pem` y te dice qué variables de entorno exportar.

PARA QUÉ SIRVE ESTO
-----------------
Si estás detrás de un proxy corporativo que inspecciona TLS (algo común en empresas),
vas a ver este error al descargar un modelo de HuggingFace:

    SSL: CERTIFICATE_VERIFY_FAILED - self-signed certificate in certificate chain

El sitio no está bloqueado: el proxy firma la conexión con su propio certificado
raíz. Ese certificado SÍ está instalado en el almacén de Windows (por eso
el navegador anda), pero NO está en el bundle de certifi, que es lo que usa `httpx`
(y por lo tanto `huggingface_hub`).

Síntoma delator, y bastante confuso: gensim descarga GloVe sin problema (usa
urllib, que en Windows sí lee el almacén del sistema) y la línea siguiente
huggingface_hub falla. No te lo estás imaginando.

ESTE SCRIPT NO DESACTIVA LA VERIFICACIÓN. Fusiona los certificados de certifi
con las raíces confiables de Windows en un único archivo PEM, y eso es lo que se
verifica. La verificación completa de la cadena sigue ocurriendo.

Alternativa de una línea, si preferís instalar un paquete más:
    pip install truststore   y   truststore.inject_into_ssl()
"""

import os
import ssl
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESTINATION = os.path.join(ROOT, ".certs", "ca_bundle.pem")


def windows_certificates():
    """Devuelve los certificados raíz confiables del almacén de Windows, en PEM."""
    if sys.platform != "win32":
        return []
    pems, seen = [], set()
    for store in ("ROOT", "CA"):
        try:
            for cert_bytes, cert_type, trust in ssl.enum_certificates(store):
                # trust es True (todos los usos) o un conjunto de OIDs; False = no confiar
                if trust is False:
                    continue
                if cert_type != "x509_asn" or cert_bytes in seen:
                    continue
                seen.add(cert_bytes)
                pems.append(ssl.DER_cert_to_PEM_cert(cert_bytes))
        except Exception as e:
            print(f"  advertencia: no se pudo leer el almacén '{store}': {type(e).__name__}: {e}")
    return pems


def main():
    if sys.platform != "win32":
        print("Este script solo hace algo en Windows. En Linux/macOS el almacén")
        print("del sistema ya se usa, o podés exportar SSL_CERT_FILE a mano.")
        return 1

    import certifi

    base = open(certifi.where(), "r", encoding="utf-8").read()
    extra = windows_certificates()
    print(f"certifi                        : {certifi.where()}")
    print(f"certificados de Windows hallados: {len(extra)}")

    os.makedirs(os.path.dirname(DESTINATION), exist_ok=True)
    with open(DESTINATION, "w", encoding="utf-8", newline="\n") as f:
        f.write(base)
        if not base.endswith("\n"):
            f.write("\n")
        f.write("\n# ---- certificados del almacén de Windows ----\n")
        f.write("\n".join(extra))

    print(f"escrito                        : {DESTINATION}  ({os.path.getsize(DESTINATION):,} bytes)")
    print()
    print("Ahora exportá estas variables ANTES de ejecutar verify_setup.py:")
    print()
    print("  # PowerShell")
    print(f'  $env:SSL_CERT_FILE = "{DESTINATION}"')
    print(f'  $env:REQUESTS_CA_BUNDLE = "{DESTINATION}"')
    print()
    print("  # Git Bash")
    print(f'  export SSL_CERT_FILE="{DESTINATION}"')
    print(f'  export REQUESTS_CA_BUNDLE="{DESTINATION}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
