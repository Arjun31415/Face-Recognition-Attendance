{
  stdenv,
  lib,
  libpqxx,
  boost,
  cmake,
  pkgs,
  cudatoolkit,
  cudaPackages,
  cudaPackages_12_0,
  pkgconfig,
  static ? false,
}: let
  stdenv = pkgs.clangStdenv;
  libx11 = pkgs.xorg.libX11;
in
  stdenv.mkDerivation {
    allowUnfree = true;
    name = "face-attendance";
    version = "1.0";
    src = ./.;

    nativeBuildInputs = [cmake];
    buildInputs = [boost libx11 cudaPackages.cudatoolkit cudaPackages.cudnn pkgconfig pkgs.openblas];

    /*
       cmakeFlags = [
      (lib.optional static "-DBUILD_STATIC=1")
      (lib.optional (!static) "-DENABLE_TESTS=1")
    ];
    */
    makeTarget = "face_attendance";
    enableParallelBuilding = true;

    installPhase = ''
      mkdir -p $out/bin ;
      cp ./face_attendance $out/bin ;
    '';
  }
