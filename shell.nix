with import <nixpkgs> {};
stdenv.mkDerivation {
  name = "fft";
  buildInputs = [
    bashInteractive
    rustup
    python3Packages.jupyter
    python3Packages.numpy
    python3Packages.matplotlib
  ];
}
