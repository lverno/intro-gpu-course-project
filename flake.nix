{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };
    cuda = pkgs.cudaPackages_13_2;
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with (pkgs // cuda); [
        cuda_nvcc
        cuda_cudart
        libcurand
        raylib
        gnumake
      ];
      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [pkgs.linuxPackages.nvidiaPackages.stable];
    };
  };
}
