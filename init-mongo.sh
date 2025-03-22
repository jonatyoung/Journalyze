#!/bin/bash
set -e

echo -e "\n==> Menunggu MongoDB siap...\n"
until mongosh --quiet --eval "db.adminCommand('ping')" > /dev/null 2>&1; do
  echo "==> MongoDB belum siap... tunggu 1 detik"
  sleep 1
done

echo -e "\n==> MongoDB siap! Memulai inisialisasi database...\n"

DB_NAME=${MONGO_DB}
DB_USER=${MONGO_USERNAME}
DB_PASS=${MONGO_PASSWORD}

echo -e "\n==> Membuat database: $DB_NAME\n"
echo -e "\n==> Dengan username: $DB_USER\n"

mongosh admin --quiet --eval "
  db = db.getSiblingDB('$DB_NAME');
  
  if (!db.getUser('$DB_USER')) {
    db.createUser({
      user: '$DB_USER',
      pwd: '$DB_PASS',
      roles: [{ role: 'dbOwner', db: '$DB_NAME' }]
    });
    print('==> User $DB_USER berhasil dibuat di database $DB_NAME');
  } else {
    print('==> User $DB_USER sudah ada');
  }
  
  db.createCollection('system_info');
  db.system_info.insertOne({
    initialization_completed: true,
    timestamp: new Date(),
    version: '1.0'
  });
  
  print('==> Koleksi awal dibuat!');
  print('==> Setup database selesai dengan sukses!');
"

echo -e "\n==> Inisialisasi MongoDB selesai!\n"